# Octopus motor control biology — literature digest (strand a)

Date: 2026-10-29. For: Breach octopus procedural animation (#104–#112, goal -> primitive -> pose stack).
Hypothesis under test: brain issues goals; arms execute a small vocabulary of stereotyped motor primitives
(reach = base->tip bend propagation; fetch = quasi-articulated pseudo-joints; crawl = push by elongation, no gait;
jet = mantle contraction + crown closing).

Read status legend: **FULL** = full text read; **ABSTRACT** = abstract/summary only (Erik should fetch);
**SECONDARY** = numbers taken from another paper's citation of it.

## Planned reading list

1. Gutfreund, Flash, Yarom, Fiorito, Segev, Hochner 1996, J Neurosci 16:7297 — organization of arm movements (reaching bend)
2. Gutfreund, Flash, Fiorito, Hochner 1998, J Neurosci — patterns of arm muscle activation in extension
3. Sumbre, Gutfreund, Fiorito, Flash, Hochner 2001, Science 293:1845 — peripheral motor program for extension
4. Sumbre, Fiorito, Flash, Hochner 2005, Nature 433:595 — fetching / pseudo-joints ("Motor control of flexible octopus arms")
5. Sumbre, Fiorito, Flash, Hochner 2006, Curr Biol 16:767 — human-like point-to-point strategy
6. Yekutieli, Sagiv-Zohar, Aharonov, Engel, Hochner, Flash 2005, J Neurophysiol 94:1443/1459 — dynamic arm model I & II
7. Levy, Flash, Hochner 2015, Curr Biol 25:1195 — crawling arm coordination
8. Levy, Nesher, Zullo, Hochner 2017 — motor control in soft-bodied animals (review) / Levy & Hochner 2017 embodied organization
9. Hochner 2012, Curr Biol — embodied view of octopus neurobiology
10. Kier & Smith 1985 (muscular hydrostats); Kier & Stella 2007 (arm musculature); Kier 2016 review
11. Huffard, Boneka, Full 2005 Science — bipedal walking; Huffard 2006 J Exp Biol — locomotion by Abdopus aculeatus
12. Wells & O'Dor / Wells 1990 — jet propulsion; Trueman & Packard 1968; Huffard 2006 swimming
13. Grasso 2008 / Tramacere et al. — suckers; Kier & Smith 1990 sucker attachment
14. Zullo et al. 2019 / Hochner & Zullo — arm nervous system, sucker-arm coordination
15. Kennedy, Buresch, ... Mather 2020 Sci Rep "Arm coordination ... in the wild" / Bidel et al.
16. Recent (2019–2026): Kennedy et al. 2020 "Octopus arms exhibit exceptional flexibility" (Sci Rep 10:20872); swimming kinematics; Grasso/Wang suckers; Hanlon reviews; sucker-led behaviours (Olson et al. 2025?) etc.

---

## Paper notes

### 1. Gutfreund, Flash, Yarom, Fiorito, Segev, Hochner 1996 — Organization of octopus arm movements
J Neurosci 16(22):7297–7307. DOI 10.1523/JNEUROSCI.16-22-07297.1996. PMC6578955.
**Read: FULL** (PMC full text, via a summarising fetch — figures' time axes not extracted).
- Reaching = a BEND formed near the arm base that travels base->tip; the arm proximal to the bend is straightish and the
  distal part is pulled along. If the arm is already extended, the reach starts by a TWIST that forms the bend, the distal
  part being pulled back first ("the movement is initiated by first creating a bend, which is formed by twisting the arm").
- The bend is ALWAYS curved DORSALLY, so the sucker rings (ventral) face the direction of travel -> suckers lead, toward the target.
- Bend midpoint moves in ONE PLANE (roughly the arm's sagittal plane), on a simple slightly curved path from the body centre to
  the target. Plane fit R^2 0.75–0.89, fit SE 0.4–0.6 cm.
- Bend-point tangential speed range 5–50 cm/s; bend-point travel 26–33 cm per reach (three animals, 300/470/700 g).
- ~70–77 % of reaches (65/84) share ONE stereotyped speed profile: phase I variable (initial, ends at a local minimum),
  phase II monotonic acceleration to peak (the most invariant phase), phase III deceleration until the bend dissolves
  near the tip. The profile collapses to one invariant curve when normalised V/Vmax vs Vmax·t/D -> the reach is ONE
  primitive scaled in amplitude and duration.
- Two arms extended in the same context have nearly IDENTICAL speed profiles (RMS speed difference 6.3 cm/s synchronous,
  6.8 consecutive, vs 13.5 controls); consecutive two-arm reaches lag 0.4–1.3 s.
- Spontaneous unconstrained extensions (no target) show the same profile (10/11); extensions sliding along wall/floor
  mostly do NOT (6/8 differ) -> the free-water reach primitive is distinct from substrate-contact exploration.
- Control hypothesis: the bend's position in space/time is the controlled variable (redundancy reduced to a point on a path).
- Game implication: one reach primitive = (target direction -> plane; peak speed; distance); suckers face forward;
  the bend dissolves at the tip at the end.

### 2. Gutfreund, Flash, Fiorito, Hochner 1998 — Patterns of arm muscle activation in reaching
J Neurosci 18(15):5976–5987. DOI 10.1523/JNEUROSCI.18-15-05976.1998. PMC6793066. **Read: FULL** (PMC, via summarising fetch).
- A WAVE of EMG travels down the arm with the bend; activation starts 200–300 ms before the bend reaches a site and peaks
  50–100 ms before it. Tonic activity sometimes persists behind the bend (proximal arm stays stiff).
- Two electrodes 2 cm apart: onset lag ~160 ms (x-corr 140 ms) -> the activation/bend front moves ~12.5–14 cm/s at that
  (proximal) location at that phase — consistent with the bend accelerating later (peak speed 550 ± 320 ms after the bend
  passes the electrode, i.e. peak velocity reached when the bend is well along the arm).
- EMG amplitude does not decay as the wave propagates; burst WIDTH is independent of speed; AMPLITUDE of the early EMG
  predicts PEAK velocity (r ~0.7–0.87) -> feed-forward: speed is set at launch, not regulated during the reach.
- Natural reach speeds 0–50 cm/s. Model arm used: 26 cm long, base radius 8.5 mm, 25 g; drag C_D ~0.31.
- Game implication: speed is a single launch parameter; proximal arm behind the front becomes a stiff rod.

### 3. Sumbre, Gutfreund, Fiorito, Flash, Hochner 2001 — Control of octopus arm extension by a peripheral motor program
Science 293:1845–1848. DOI 10.1126/science.1060976. **Read: ABSTRACT ONLY** (+ Semantic Scholar TLDR) — Erik to fetch.
- Arm extensions with bend propagation can be evoked mechanically (touching the skin) or electrically (axial nerve cord)
  in arms DENERVATED from the brain, with kinematics "almost identical to normal behavior".
- => the reach primitive is generated in the arm itself; the brain only triggers it (and chooses direction/amplitude).
- (Recalled, unverified — check in full text: velocity profiles of denervated-arm extensions matched the normalised
  intact profile; extension also evoked in amputated arms tens of minutes after severing.)
- Game implication: strong support for "brain issues goal + trigger; arm runs the primitive locally".

### 4. Sumbre, Fiorito, Flash, Hochner 2005 — Motor control of flexible octopus arms (fetching)
Nature 433:595–596. DOI 10.1038/433595a. **Read: ABSTRACT ONLY** — Erik to fetch (brief communication, has supp. video).
- When transferring a held object (food) to the mouth, the arm reconfigures into a STIFFENED, ARTICULATED, quasi-jointed
  structure ("vertebrate-like strategy") — three pseudo-joints, rotation about them carries the object to the mouth.
- (Secondary, to verify: the two proximal segments — "upper arm" and "forearm" — are of roughly EQUAL length, as in a
  human arm; see #5 and later notes.)

### 5. Sumbre, Fiorito, Flash, Hochner 2006 — Octopuses use a human-like strategy for point-to-point arm movements
Curr Biol 16(8):767–772. DOI 10.1016/j.cub.2006.02.069. **Read: ABSTRACT ONLY** (Cell open archive PDF exists —
http://www.cell.com/article/S0960982206012747/pdf — but returned HTTP 403 to my fetcher; Erik should fetch).
- Fetch: three dynamic joints — proximal (near base), medial ("elbow"), distal ("wrist", at the grasp point);
  rotation around them brings the object to the mouth.
- Mechanism: TWO waves of muscle activation propagate toward each other — one from the grasp point (distal) inward,
  one from the base outward — and their COLLISION point sets the medial joint. So segment lengths adapt to where along
  the arm the object is held: grasp near the tip -> long segments; grasp near the base -> short ones.
- Kinematic invariants are found at the JOINT level (joint-angle profiles), not the end-effector -> control in joint space.
- Caveat (blog review): animals trained to extend and wait for food; generality across natural feeding uncertain.
- Game implication: fetch = (grasp station s_g) -> elbow at ~s_g/2, i.e. two equal segments; rigid links + 3 hinge angles.

### 6. Yekutieli, Sagiv-Zohar, Aharonov, Engel, Hochner, Flash 2005 — Dynamic model of the octopus arm I & II
J Neurophysiol 94:1443–1458 (I, DOI 10.1152/jn.00684.2004) and 94:1459–1468 (II, DOI 10.1152/jn.00685.2004).
**Read: ABSTRACTS ONLY** (journals.physiology.org returned 403) — Erik to fetch both.
- 2D multi-segment model; each segment has longitudinal + transverse muscle and CONSTANT VOLUME (muscular hydrostat:
  thinning = lengthening). Includes gravity, buoyancy, drag.
- Reach is reproduced by ONE wave of activation moving at constant velocity along the arm: a STIFFENING front that
  pushes a passive bend ahead of it. Bend = boundary between the stiff (proximal) and the still-limp (distal) arm.
- Only TWO control parameters fully specify the extension: activation amplitude and wave travel time. The same
  kinematics result for any amplitude above a minimum; extra amplitude buys stiffness (perturbation resistance) without
  changing the kinematics -> kinematics and stiffness are independently controllable.
- Perpendicular drag coefficient ~50x tangential; during a reach only a short piece of arm (the bend) is broadside to the
  flow, minimising drag -> explains WHY the bend primitive exists in water (unrolling a "whip" through water).
- Game implication: reach = a stiffness front s_f(t) travelling base->tip; arm behind it straight (rigid), arm ahead of it
  trails passively (dragged, curled back). Two parameters: travel time T and amplitude/stiffness.

### 7. Levy & Hochner 2017 — Embodied organization of Octopus vulgaris morphology, vision, and locomotion (review + new data)
Front Physiol 8:164. DOI 10.3389/fphys.2017.00164. PMC5368235. **Read: FULL** (PMC full text).
- Reach = combination of primitives: bend propagation + INDEPENDENT elongation of the arm segment proximal to the bend.
  Central control of reaching reduced to 3–4 DOF: 2 for the arm-base direction, 1 for the stiffening wave, possibly
  1 for elongation/straightening. Amputated arms generate the same extension on a short electrical train (Sumbre 2001).
- Fetch: arm reshaped into a three-segment skeleton-like structure, PROXIMAL AND MEDIAL SEGMENTS OF SIMILAR LENGTH, the
  food held by the DISTAL segment "using a group of suckers (serving as a hand)". The food is brought to the mouth by
  rotating the PSEUDO-ELBOW between proximal and medial segments. Equal segments make the hand land on the mouth, which
  sits at the centre of the circle of arm bases. After contact with the target, one wave runs proximally from the contact
  site, the other distally from the arm base; the elbow forms where they collide. Joints are rebuilt for every fetch.
- Brain role: activation and SCALING of peripheral programs plus few-DOF visual/tactile adjustment (Zullo 2009: the
  higher motor centres are not somatotopic).
- Crawling: moment-to-moment choice of WHICH arm(s) to recruit. A GROUP OF SUCKERS on the chosen arm adheres to the
  substrate as an anchor; a stereotyped ELONGATION of a PROXIMAL segment generates thrust (push, not pull). Each arm has ONE
  predefined pushing direction set by its position around the body; the instantaneous crawl direction is the VECTOR SUM
  of the active arms' pushing directions; active arms apply roughly equal forces.
- No order in the stepping records; FFT of crawl velocity shows no characteristic frequency (unlike Drosophila or stick
  insect) — the 0.3 Hz peak in one record is just the window (a 6 s bout). Same for "walking" (body raised): no order, no
  rhythm; walking needs >= 2 arms in contact for support.
- HEAD STAYS HORIZONTAL during all locomotion (the eye-to-eye axis ~level; statocysts sit in the cranium). Achieved by
  controlling the STIFFNESS/length of arms between contact points and the head base (1 DOF per contacting arm), i.e. the
  arms' contact geometry sets the head height — physical feedback, not a CPG.
- Can crawl in ANY direction regardless of facing; facing is controlled independently.
- Brain: ~500 million neurons; ~2/3 in the arms (with few fibres connecting the compartments).
- Game implication: crawl controller = pick arms whose pushing direction has positive dot with the desired velocity, plant
  a sucker group, extend the proximal segment; no gait table. Keep head level by arm-length adjustment.

### 8. Levy, Flash, Hochner 2015 — Arm coordination in octopus crawling involves unique motor control strategies
Curr Biol 25(9):1195–1200. DOI 10.1016/j.cub.2015.02.064. **Read: ABSTRACT + press release + the authors' own
2017 summary (#7) ONLY** — Erik to fetch (it has supplementary movies; EurekAlert hosts two clips, "Crawling Octopus
1 of 2 / 2 of 2", eurekalert.org/multimedia/650114 and /650113).
- Crawl in any direction relative to body orientation; body orientation and crawl direction controlled monotonically
  and independently. No rhythmic pattern in limb coordination.
- Thrust by "pushing-by-elongation": an active arm shortens and elongates "like an inchworm" (press quote) — the arm's motion
  when active is always the same. The central controller only chooses which arms to recruit.
- Numbers still missing (need the full text): push duration, elongation strain per push, how many arms active at once,
  crawl speed, body sizes. (From memory, NOT verified: ~1–4 arms active simultaneously, pushes of order 1–2 s, arms in
  front and to the side of the direction of motion recruited — check.)

### 9. Hanassy, Botvinnik, Flash, Hochner 2015 — Stereotypical reaching involves both bend propagation and arm elongation
Bioinspir Biomim 10(3):035001. DOI 10.1088/1748-3190/10/3/035001. **Read: ABSTRACT ONLY** — Erik to fetch.
- Reaches also ELONGATE the proximal arm (base -> bend). Average strain along the arm at end of movement: -0.12 (shortening)
  up to 1.8 (!), mean 0.57 ± 0.41 (n = 64, 4 animals); with target and animal kept still 0.64 ± 0.28 (n = 30, 2 animals).
  So a reaching arm typically ends ~1.6x its resting length over the extended part.
- Two kinds of reach: (a) bend starts at the base and propagates to the tip; (b) the bend is formed/present DISTALLY and
  the reach is mainly elongation + straightening of the segment proximal to the bend (elongation correlates with
  bend-target distance only in type b).
- Elongation and bend propagation are separate primitives, combined in varying proportions.
- The Yekutieli model cannot produce this elongation with physiologically plausible muscle ratios (needs
  longitudinal/transverse force < 0.0016) -> the model's activation assumptions need revising.

### 10. Richter, Hochner, Kuba 2015 — Octopus arm movements under constrained conditions
J Exp Biol 218:1069–1076. DOI 10.1242/jeb.115915. **Read: ABSTRACT ONLY**.
- Arm put through a hole in a Perspex box (N = 6): reaches either start a stereotyped bend propagation AT THE HOLE (the
  hole acts as a new arm base) or are "waving-like" movements with no bend propagation.
- Fetch through the hole: no complete pseudo-joint fetch; instead "pull-in" — the arm is hauled in through the hole.
  Authors name "arm pulling" a separate motor primitive.
- Game implication: primitive start point is relocatable (the bend can launch at any station); "pull-in" is a fallback
  fetch when articulation is blocked.

### 11. Zelman, Titon, Yekutieli, Hanassy, Hochner, Flash 2013 — Kinematic decomposition and classification of octopus arm movements
Front Comput Neurosci 7:60. DOI 10.3389/fncom.2013.00060. PMC3662989. **Read: FULL** (PMC, via summarising fetch; numbers mostly in figures).
- 3D reconstructions, 4 O. vulgaris (200–470 g), 50 fps, 16 °C. 85 movements: 60 extensions + 25 "pre-extension"
  (initialisation movements that precede the bend propagation).
- Curvature and torsion along the arm (arm index x time) each decompose into 3 Gaussian "kinematic units"; extensions fall
  into 3 prototypes, pre-extensions into 1. Some prototypes are a bend propagating "from the MIDDLE of the arm toward
  the tip" (cf. Hanassy type b); during the extension "the arm configuration became relatively planar" and
  "the proximal section of the arm elongates".
- Torsion (twist) is a real, separate component — matches 1996's "bend formed by twisting".
- Game implication: a reach is preceded by a short preparatory twist/curl (pre-extension); planar during the main phase.

### 12. Zullo, Eichenstein, Maiole, Hochner 2019 — Motor control pathways in the nervous system of Octopus vulgaris arm
J Comp Physiol A 205:271–279. DOI 10.1007/s00359-019-01332-6. PMC6478645. **Read: FULL** (PMC, via summarising fetch).
- Arm nerve cord swells into a ganglion opposite EACH sucker; plus five intramuscular nerve cords and the sucker ganglia.
- ~380,000 motor neurons per arm; ~1,500 motor neurons per mm of arm.
- Cerebrobrachial-tract axons run >= 30 mm along the arm and drive motor neurons "en passant"; conduction ~250–300 cm/s
  (i.e. ~1 ms per 3 cm — command transit is effectively instantaneous on animation timescales; the 10–50 cm/s bend speed is
  a muscle/mechanics wave, not a conduction delay).
- "Octopuses can initiate stereotypical movements practically at any point along the arm" -> the primitive launch station
  is a parameter (consistent with Hanassy type b, Richter's hole).

### 13. Huffard 2006 — Locomotion by Abdopus aculeatus: walking the line between primary and secondary defenses
J Exp Biol 209:3697–3707. DOI 10.1242/jeb.02435. **Read: FULL** (author-hosted PDF, birdsheadseascape.com, text-extracted).
Species: Abdopus aculeatus, 11 adults, ML 24–43 mm, 11–62 g; arms long relative to body. 30 fps overhead video.
Mode definitions (useful verbatim for a state machine):
- JET: water expelled through funnel pointed AWAY from the mantle tip -> animal moves BACKWARD, mantle first; arms TRAIL,
  bear no weight; held tightly together (most common), in a V, or spread laterally. Elongate body, straight arms together,
  skin SMOOTH and usually PALE (papillae flattened to cut drag).
- SWIM: funnel pointed back toward the mantle -> animal moves FORWARD, head/eyes/arm crown first; no part on the bottom;
  with or without undulation of body and arms. Postures range from HEAD-RAISED (arms close to and under the body) to
  DORSOVENTRALLY COMPRESSED (head and mantle lowered into the arms' plane, arms held forward and to the side); flat (DVC)
  swimming is 23–59 % faster (lift, "flatfish").
- CRAWL: sucker edge of MORE THAN TWO arms pushes and pulls along the bottom in any direction; each arm contacts the bottom
  at MULTIPLE points. Most varied postures/colours of any mode. Variants: sprawled; upright (mantle pointed up, arms dragged
  under/behind); conical (rolling along the sucker edge of arm pairs III and IV, or several arms in unison pushing/pulling in
  JERKY movements, arms curved back); ball crawling ("moving rock").
- BIPEDAL WALK: alternating movements of ONE pair of arms, pair IV (ventral-most), ROLLING along the sucker edge
  (a tread-like roll, "large footprint"), usually backward or obliquely backward; other six arms coiled/raised (algae disguise)
  or tucked under. Third/fourth arm sometimes touches briefly ("multi-armed walk"). Also forward walking on stiffened bent
  alternate arms IV, and hopping backward on pair IV with jetting. Some walkers lean back and seem dynamically stable.
Speeds (mean ± SD; max):
- Jet 1.73 ± 0.80 BL/s, 45.3 ± 19.4 cm/s (max 3.29 BL/s, 70.1 cm/s); n = 62 bouts, 10 animals.
- Swim 1.27 ± 0.32 BL/s, 10.8 ± 6.4 cm/s (max 3.06 BL/s, 20.2 cm/s); n = 45, 9 animals.
- Crawl 0.62 BL/s, 1.54 ML/s, 7.3 ± 2.1 cm/s (max 1.94 BL/s, 4.31 ML/s, 21.3 cm/s); n = 86, 10 animals.
- Bipedal walk 1.34 ± 0.19 BL/s, 13.1 ± 11.5 cm/s (max 2.25 BL/s, 19.0 cm/s); n = 23, 5 animals.
- Multi-armed walk 0.80 BL/s (max 1.39 BL/s, 10.6 cm/s) [cm/s mean printed as 0.68 — typo in paper].
- O. vulgaris (Wells 1987/1990 via this paper): jet max ~100 cm/s, crawl average ~9 cm/s. A. marginatus walks ~14 cm/s, 2.6 ML/s.
- Absolute speed NOT correlated with body mass in any mode -> do not scale locomotion speed with octopus size.
- Jetting body is so elongated that the whole animal takes about as long to pass a point as when walking/swimming.
- "Medusoid jetting": arms and web opened and closed like an umbrella to supplement the jet (Forsythe & Hanlon 1988);
  used in sustained chases. Large O. vulgaris, O. californicus, E. dofleini jet with arms spread laterally (lift).
Supplementary movies (JEB DC1): medusoid jetting, walking.

### 14. Kennedy, Buresch, Boinapally, Hanlon 2020 — Octopus arms exhibit exceptional flexibility
Sci Rep 10:20872. DOI 10.1038/s41598-020-77873-7. PMC7704652. **Read: FULL** (PMC).
- O. bimaculoides, 10 animals, 400–700 g, lab tanks (free, single hole, multi-hole barriers), objects/food on rods.
- 4 basic deformations: BEND (oral, aboral, inward, outward), TORSION (CW, CCW), ELONGATION, SHORTENING (+ stiffening,
  not scorable on video). Muscle basis: elongation = transverse muscle; shortening = longitudinal; stiffening = co-activation;
  bending = longitudinal on the inner side + transverse resisting compression; torsion = helical oblique muscles (both hands).
- 16,563 deformations in 120 min of video (~138 per minute across arms). Bends 11,074 (67 %), elongation 2,723 (16 %),
  torsion 1,426 (9 %), shortening 1,340 (8 %).
- Anterior arms (pairs 1–2) much more active: 8,227 vs 2,847 bends; elongation 1,887 vs 836. ORAL bending is the LEAST common
  bend direction (1,347 of 11,074).
- By region: PROXIMAL third mostly elongates/shortens (bends only 1,464 of 3,537 proximal deformations, 41 %); MEDIAL most
  active overall (7,775 deformations, 71 % bends); DISTAL 5,251, 78 % bends. Proximal torsion in arm pair 1 is rare.
- Deformations happen MORE OFTEN WITHOUT sucker contact than with -> suckers are not needed to drive arm shape.
- Elongation/shortening judged by sucker SPACING and arm THICKNESS (constant volume: longer = thinner) — a visual cue a
  renderer must reproduce.
- Functional reading: proximal = aim (direction + elongation), medial = connect/most mobile, distal = touch/sense.
  "One primary function of an octopus arm is to bring the suckers to bear on a surface": reaching arms ROTATE so the oral
  sucker face meets the target. Prior study: reaches often hit the target with the CENTRAL THIRD of the arm.
- Under constraint, fetch is a straight PULL-IN (cf. Richter 2015).

### 15. Bennice, Buresch, Grossman, Morano, Hanlon 2025 — Octopus arm flexibility facilitates complex behaviors in diverse natural environments
Sci Rep 15:31875. DOI 10.1038/s41598-025-10674-y. PMC12426218. **Read: FULL** (PMC, via summarising fetch; the 12-action
ethogram definitions are only in a figure image I could not read — Erik should look at Fig. 2).
- WILD footage: O. vulgaris/O. briareus/O. insularis-type shallow-water octopuses, 25 one-minute clips, 25 individuals, 6 sites
  (Spain + Caribbean), 1–10 m depth, mantle length 10–17 cm.
- 12 ARM ACTIONS (an empirical action vocabulary!): reach 19 %, raise 18 %, lower 17 %, tuck 14 %, curl 10 % (top five = 78 % of
  3,907 actions); rarer: parachute, roll, grasp, stilt, tiptoe, push (+1 unnamed in text).
- Front arms do 64 % of actions overall (reach 63 %, curl 71 % anterior); posterior arms dominate STILT (63 % posterior) and
  ROLL (71 % posterior) -> back arms carry/locomote, front arms explore. No left/right bias in the wild.
- Behaviours and number of distinct actions they use: crawl 11/12, crawl-speculative bottom searching 10, moving seaweed 8,
  parachute attack 8, backward swimming 3, grope 3, fetch 2.
- Deformations in the wild: bend 70 %, elongate 22 %, shorten 6 %, torsion 2 % (n = 6,781). Distal 47 %, medial 35 %,
  proximal 18 %; proximal elongates most, distal bends most; torsion never proximal.
- Multiple actions occur simultaneously on one arm and across adjacent arms.
- Game implication: this is the closest published "action vocabulary" for wild arm animation; the front/back split is a direct
  rule (front 2 pairs explore/reach, back 2 pairs stilt/roll/support).

### 16. Wells, Duthie, Houlihan, Smith, Wells 1987 — Blood flow and pressure changes in exercising octopuses (O. vulgaris)
J Exp Biol 131:175–187. DOI 10.1242/jeb.131.1.175. **Read: ABSTRACT ONLY** — Erik to fetch (JEB full text paywalled to my fetcher).
- Jet propulsion is accompanied by CARDIAC ARREST (mantle pressure stops venous return); oxygen debt sustainable ~22 ml O2/kg;
  "jet-propelled movement is impossible for distances of more than a few metres".
- Game implication: jetting is a short BURST (a few metres, a few strokes), then the animal must crawl/rest — octopuses are
  crawlers that jet to escape, not cruising swimmers. A "swim everywhere" octopus is wrong.
- (Wells 1990, Can J Zool 68:815 "Oxygen extraction and jet propulsion in cephalopods", cited by Huffard 2006: O. vulgaris
  jet max ~100 cm/s, crawl average ~9 cm/s. NOT READ.)

### 17. Arm-swimming / sculling — Sfakiotakis, Kazakidi, Tsakiris 2015 (and their ICRA 2012/2013 papers)
Sfakiotakis M, Kazakidi A, Tsakiris DP 2015 "Octopus-inspired multi-arm robotic swimming", Bioinspir Biomim 10:035005,
DOI 10.1088/1748-3190/10/3/035005 (author manuscript at strathprints.strath.ac.uk/60917, 24 MB — too big for my fetcher).
Biological background: Sfakiotakis et al. ICRA 2013 "Octopus-inspired eight-arm robotic swimming by sculling movements"
(projects.ics.forth.gr/cvrl/octopus/publications/ICRA13_2200_MS.pdf — TLS error for my fetcher).
**Read: ABSTRACT + search-engine snippets ONLY (SECONDARY)** — Erik to fetch the ICRA 2013 PDF for the biology section.
- Besides siphon jetting, octopuses "arm-swim": the arms (and web) open SLOWLY (recovery stroke) and close RAPIDLY inward
  (power stroke) while rotating about their bases; thrust comes in the power stroke (= "medusoid" swimming of Huffard).
- Recovery/power duration ratio ~2.5 ± 0.5, from video of 2 wild Abdopus aculeatus and 5 captive O. vulgaris (snippet,
  verify). So a stroke is ~70 % slow open, ~30 % fast close.
- CFD (Kazakidi et al. 2015, DOI 10.1080/10255842.2014.900757): fast-power/slow-recovery strokes give net thrust,
  harmonic (symmetric) strokes do not -> a symmetric sine "flapping" of the arms is physically and visually wrong.
- Game implication: jet primitive = mantle inflate (slow) + contract (fast) coupled to the arm crown open (slow) / close
  (fast); the arm closing should be the fast phase, ratio ~1:2.5.

### 18. Grasso 2008 — Octopus sucker-arm coordination in grasping and manipulation
Amer Malacol Bull 24:13–23. DOI 10.4003/0740-2783-24.1.13. **Read: FULL** (BHL/BioStor scan, text-extracted).
O. bimaculoides, arms 15–20 cm, 150–300 suckers per arm (tip suckers < 1 mm), two staggered rows; suckers get smaller distally;
adhesion force ∝ sucker size and is additive across suckers. "About 300 suckers" typical (Voight 1993).
- Sucker anatomy (for the rig): two chambers — infundibulum (rim, conforms to the surface and seals) and acetabulum (expands
  to make suction); EXTRINSIC muscles rotate the sucker rim in virtually any plane about the arm axis, WITH whatever it holds
  -> suckers are small manipulators, not just suction cups. Rim receptors ~10^4 per sucker. Each sucker has its own ganglion.
- Suckers explore WITHOUT sealing: on a stationary arm they extend/retract/rotate to follow contours and edges with only the rim.
- On a glass wall, only 18 ± 14 % of visible suckers are attached at any instant (range 0–39 %); NEVER all suckers of an arm,
  even when the arm is motionless. -> render most suckers free/touching, a minority sealed.
- "ARM WALK": the arm translates sideways across a surface with suckers of the two ROWS attaching in ANTI-PHASE
  (alternating left/right row), groups of 6, 10 or 15 adjacent suckers coordinated, single alternation up to ~8 cycles over
  several seconds (~1 Hz order from the figure: 8 cycles in "several seconds"), leading and trailing pairs advancing.
- FIXED GRIP: one row (one side) of suckers held attached for up to 25 s (horizontal stripe in the raster).
- ANCHOR POINTS: 3–5 adjacent attached suckers; the arm distal to the anchor then moves by a WHIP-like wave running from the
  anchor distally, distal suckers free -> the bend-propagation primitive can launch from any anchored station.
- PULL-UP ("octopus lift"): arm projected forward/up, distal suckers attach, arm SHORTENS pulling the body to the anchor.
  First a widespread probing attachment of 12–15 adjacent suckers on alternating sides, then narrowing to 2–3 adjacent
  suckers as the fixed point during the pull. 16 s for two repetitions (~8 s per pull cycle in that climb).
- 4–8 simultaneously attached suckers anchor an arm that lifts a portion of itself off the surface.
- Dome-raising (lift a 10–14 N weighted dome): animal drapes the WEB over the object, mouth over its apex, arms down the sides
  onto the floor; it raises it by LENGTHENING the arm segments between suckers fixed on the dome and suckers fixed on the floor
  (pushing off the floor, not lifting with the arm). Always used the MIDDLE third of the arm (12/12 trials), not the big
  proximal suckers. Used 10.2 ± 1.5 suckers (weak) vs 15.7 ± 2.6 (strong force).
- Octopuses rarely WRAP an arm around an object to grasp; wrapping looks like arm momentum continuing after abrupt contact.
  Power grasp = many suckers on one surface. Squid-style "shoot arm, stick distal suckers, pull in by shortening" exists too.
- Byrne et al. 2006 (cited): O. vulgaris first contacts a target with the MIDDLE of the arm.

### 19. Gendreau, Boucaud, Buresch, Rooney, Morris, Hanlon 2025 — How octopuses use and recruit additional arms to find and manipulate visually hidden items
Biol Open 14(7):bio062011. DOI 10.1242/bio.062011. PMC12309893. **Read: FULL** (PMC, via summarising fetch).
- O. bimaculoides, 14 animals, 117 trials; search a rock-covered opaque dome with 4 openings for a hidden item.
- FIRST arm in is anterior 75 % (88 vs 29), no single preferred anterior arm. Exploratory SWEEPING in 94 % of approaches.
- Recruitment: further arms join in 55 % of approaches (394 events); the NEIGHBOUR arm (A+1) 44 % of recruitments, A+2..A+4 56 %.
  >= 3 arms in sequence in 29 %; commonest sequences A→A+1→A+1 (~26 %) and A→A+1→A+2 (13 %) — recruitment spreads to
  ADJACENT arms.
- Simultaneous arms inside the dome: 2 arms 5 %, 3 arms 27 %, 4 arms 31 %, 5 arms 31 %, 6 arms 4 %; NEVER all eight.
- Removable item extracted in 90 % of trials.
- Game implication: manipulation goals recruit a neighbourhood of arms around the first arm, typically 3–5 arms, rest support.

### 20. Gribkova, Cui, Gillette 2025 — Octopus "hypnosis": tonic immobility for studying local sensorimotor responses and arm-sucker coordination
Biol Open 14(12):bio062082. DOI 10.1242/bio.062082. PMC12714133. **Read: FULL** (PMC, via summarising fetch; preliminary data).
- Under tonic immobility, touching one arm gives only LOCAL responses (no other arms recruited).
- Chemotactile plume on a sucker: the sucker EXTENDS TOWARD (attractive) or RETRACTS AWAY (aversive); neighbouring suckers are
  recruited; sometimes the arm reaches or retracts. Single suckers orient precisely to the bearing of a rim stimulus.
- Two simultaneous stimuli: suckers pick ONE (winner-take-all), not the average (preliminary).
- Game implication: sucker-level reflex layer = orient toward stimulus, recruit neighbours, escalate to arm reach.

### 21. Huffard, Boneka, Full 2005 — Underwater bipedal locomotion by octopuses in disguise
Science 307:1927. DOI 10.1126/science.1109616. **Read: ABSTRACT + SICB conference abstract + search snippets of a scan
(SECONDARY)** — Erik to fetch the paper AND its Supporting Online Material (stride kinematics, movies).
- O. marginatus (coconut disguise) and Abdopus aculeatus (algae disguise) walk on the VENTRAL pair (arms IV), alternating,
  each arm tip ROLLING along its sucker edge ("like a tank tread"/belt), pushing the animal BACKWARD; the other six arms
  are used for camouflage (wrapped/raised).
- Each arm on the sand > half the stride (duty factor > 0.5 for both species -> walking, not running); snippet of the scan
  gives per-arm duty factors 0.69 (right) and 0.91 (left) — verify. Speed of O. marginatus <= 0.1 m/s (~2.6 ML/s).
- Contrast drawn by the authors: ordinary crawling = several arms sprawled, pushed and pulled by suckers "in no regular sequence".

### 22. Mather 1998 — How do octopuses use their arms?
J Comp Psychol 112:306–316. DOI 10.1037/0735-7036.112.3.306. **Read: ABSTRACT ONLY** — Erik to fetch (key descriptive taxonomy).
- A taxonomy of arm movement components: movements of the arm itself, of the suckers and their STALKS, relative position of
  arms, and the WEB between them. Within one arm, components combine into behaviours.
- Whole-animal arm behaviours split into POSTURES (all arms + web spread into a 2D surface positioned in 3D — e.g. web-over/
  parachute) and ACTIONS (more or less coordinated, several to all arms).
- Control principles proposed: radial symmetry, relative EQUIPOTENTIALITY of arms, relative INDEPENDENCE of each arm,
  SEPARABILITY of components within an arm.
- Mather & Alupay 2016 "An ethogram for benthic octopods" (J Comp Psychol 130:109–127, DOI 10.1037/com0000025, ABSTRACT ONLY):
  six structure sets — mantle-funnel, arms, sucker-stalk, skin-web, head, mouth — combine into behavioural units (postures,
  self-directed movements like head bob and grooming, whole-animal jetting and crawling). Useful as the master action list.

### 23. Byrne, Kuba, Meisel, Griebel, Mather 2006 — Does O. vulgaris have preferred arms? / Octopus arm choice is strongly influenced by eye use
J Comp Psychol 120:198–204 (DOI 10.1037/0735-7036.120.3.198) and Behav Brain Res 172:195–201 (DOI 10.1016/j.bbr.2006.04.026).
**Read: ABSTRACTS ONLY.**
- Strong preference for ANTERIOR arms to reach for and explore objects (task division anterior/posterior); 4/8 lateral bias.
- Arm choice is visually opportunistic: the arm used to touch an object is the one on the line between the LOOKING EYE and the
  object. First contact is usually made with the CENTRAL THIRD of the arm. Next arm to touch is most likely a NEIGHBOUR.
- Game implication: goal arm selection = arm whose base lies nearest the eye→target line, contact with mid-arm, recruit neighbours.

### 24. Flash & Hochner 2005 — Motor primitives in vertebrates and invertebrates
Curr Opin Neurobiol 15:660–666. DOI 10.1016/j.conb.2005.10.011. **Read: ABSTRACT ONLY.**
- Framework: movements composed of elementary building blocks; repertoire spanned by transformations (scaling, rotation,
  time-warping) of primitives and their combination by syntactic rules; primitives can exist at neural, dynamic and kinematic
  levels. The authors' own framing of reach/fetch as octopus motor primitives. Directly supports the goal->primitive design.

### 25. Kier & Stella 2007 — The arrangement and function of octopus arm musculature and connective tissue
J Morphol 268:831–843. DOI 10.1002/jmor.10548. **Read: ABSTRACT ONLY.** (also Kier & Smith 1985 Zool J Linn Soc 83:307–324,
"Tongues, tentacles and trunks: the biomechanics of movement in muscular-hydrostats", not read)
- Three muscle orientations around the axial nerve cord: TRANSVERSE (-> elongation, since volume is constant), LONGITUDINAL
  (-> shortening; unilateral -> bending), OBLIQUE helices both hands (-> torsion). Proportions constant along the arm though
  it tapers. Bending = unilateral longitudinal + transverse resisting diameter increase; flexural stiffness = co-contraction
  of transverse + longitudinal; torsional stiffness = both helices together.
- Game implication: per-segment state (length strain, curvature vector, twist, stiffness) is the true DOF set, and volume
  conservation couples length to thickness (render thickness ∝ 1/sqrt(length strain)).

### 26. Katija, Huffard, Roberts, Daniels, Erickson, Klimov, Ruhl, Sherman 2025 — In situ light-field imaging of octopus locomotion reveals simplified control
Nature (6 Aug 2025). DOI 10.1038/s41586-025-09379-z. PMC12545212. **Read: FULL** (PMC, open access CC BY 4.0;
numbers in Extended Data figures not extracted). THE most important recent crawling paper — wild, 3D, many strides.
- Muusoctopus robustus, 3,200 m deep, Davidson Seamount "Octopus Garden", 1.6 °C; head widths 52–90 mm. Strides counted:
  O1 25, O3 73, O14 62, O15 67; O15 tracked crawling FORWARD over rough rock for ~12 min.
- ARM ROLES: arm pair 1 (dorsal) used LESS for support and mainly to PROBE the upcoming substrate; arm pairs 2–4 bear weight.
  Dorsal arms: longer arm cycles, smaller contact duty cycles, more variable; ventral arms (3, 4): shorter cycles, larger duty
  cycles, little left/right difference.
- Number of arms in contact: on average 3 ± 1 at any moment; arms on BOTH sides in support in 73 % of frames; the count cycles
  between 2 and 6 over 2–3 min periods. Order shows CYCLICAL contact from VENTRAL to DORSAL arms, "with little to no occurrence
  of arms 1 after a consistent arm pair". The authors say the cyclical bilateral use "might hint at a previously undescribed
  central pattern generator" -> this partly CONTRADICTS Levy et al. 2015 (no rhythm) for forward crawling on rough terrain in the wild.
- THE FORWARD-CRAWL STRIDE (a joint-like primitive, distinct from the bend-propagation reach and from the rolling bipedal walk):
  1. the arm places a MEDIAL portion on the substrate — the "FOOT"; 2. a region of HIGH CURVATURE (a bend, a joint) is held
  just proximal to the foot while the body moves forward; 3. the proximal weight-bearing part first SHORTENS then ELONGATES
  (strain concentrated in the third of the arm just proximal to the foot); 4. the foot stays put until the proximal part reaches
  MAXIMAL STRAIN, then the suckers lose purchase and the foot is pulled off as the body continues; 5. after release the proximal
  arm CONTRACTS and the arm swings UNDER the body for the next cycle. The part DISTAL to the foot hardly changes during the cycle.
- Before release the animal "straightened and stretched the proximal section" (L2, L3 of O15).
- Weight-bearing and non-weight-bearing parts of the same arm are strained very differently — joint-like localisation.
- Speculated mechanics: underwater spring-loaded inverted pendulum + centre-of-mass inside the support polygon (3 arms).
- Introduction notes a recent field study found primitives "more flexible when used by free-living individuals" in fast
  dynamic tasks (the arm slap, #27).
- Game implication: the crawl primitive per arm = plant mid-arm foot, keep a knee-like bend at the foot, push by proximal
  elongation, release at max strain, recover by contracting and swinging under the body; ~3 feet down; front pair probes.

### 27. Alupay, Mather, Iskarous 2023 — A syntactic analysis of a complex motor action: the octopus arm "slap"
Mar Biol 170:99. DOI 10.1007/s00227-023-04243-y (preprint Research Square DOI 10.21203/rs.3.rs-2575172/v1).
**Read: SEARCH SNIPPETS / ABSTRACT ONLY** — Erik to fetch.
- Wild Abdopus sp. slapping a blenny (deterrence, not capture; behaviour first described by Mather 1992 in O. vulgaris);
  19 slaps, 48 fps, one octopus.
- Two primitives: ROTATION of the arm (about its base) and TRANSLATION, combined SERIALLY and IN PARALLEL across different arms,
  aimed at a moving target with sensory feedback; STIFFENING matters for both ("flexible rigidity": arm stays flexible but only a
  few DOF controlled).
- Game implication: a "strike/slap" = stiffened arm swung about its base (rotation) plus base translation; syntax = sequence/overlap.

### 28. Olson, Schulz, Ragsdale 2024/2025 — Neuronal segmentation in cephalopod arms
bioRxiv 10.1101/2024.05.29.596333 (PMC11160704); published Nature Communications 2025 (not checked). **Read: ABSTRACT ONLY.**
- Axial nerve cord is SEGMENTED: 15 neuronal segments per pair of suckers; nerves from two segments (three septa) innervate the
  arm musculature; "suckerotopy" of sucker nerves. Axial nerve cords together hold 4x the neurons of the central brain.
- Game implication: the natural control unit along the arm is the sucker pair — a per-sucker-station controller grain is biologically apt.

### 29. Hale 2025 — Octopus as a comparative model for understanding the neural control of limb movement and limb-based behaviors
Curr Opin Neurobiol 2025:102982. DOI 10.1016/j.conb.2025.102982. **Read: ABSTRACT ONLY.**
- Arm nerve cords connect to each other at their bases through a NERVE RING; signals pass arm to arm. Connection to the brain is
  NECESSARY for locomotion, while arm movements can be triggered locally by naturalistic mechanosensory input.
- Game implication: inter-arm coordination (crawl) is central/ring-mediated; single-arm reflexes are local.

### 30. Weidig, Bush, Jimenez, Pelled, Bush 2024 — 3D octopus kinematics of complex postures
PLoS One 19:e0303608. DOI 10.1371/journal.pone.0303608. PMC11135735. **Read: FULL** (via summarising fetch). Minor value.
- O. bimaculoides, 150 Hz motion capture, 9 isolated arms + 1 live arm. Live-arm curvature in "simple bending" ~0.043 mm^-1
  proximal (radius ~23 mm) / 0.016 mm^-1 distal (~63 mm); straight-arm residual curvature radius 140–250 mm. "Complex
  bending": distal segment nearly flipped (internal rotation −145°). n = 1 animal — anecdotal.

### 31. Bidel, Bennett, Wardill 2022 — Octopus bimaculoides' arm recruitment and use during visually evoked prey capture
Curr Biol 32(21):4727–4733. DOI 10.1016/j.cub.2022.08.080 (erratum 4780–4781). Data: Dryad doi:10.5061/dryad.j9kd51cgg,
Zenodo 7659059. **Read: ABSTRACT + press release + the dataset README (scoring definitions) ONLY** — Erik to fetch (has videos).
(This is presumably the "Bidel" work Erik remembered; Wardill lab, U Minnesota.)
- 628 successful predatory strikes, 10 O. bimaculoides.
- Approach modes: AMBUSH (lie and wait), PURSUIT (prey flees, octopus follows), STALKING (slow gentle approach, then sudden assault).
- Strike modes (an empirical attack vocabulary): REACH (bend propagation base->tip, 1–3 arms, sequential); WAVING (arms sway
  around the prey with torsion, no bend propagation, "seemingly undirected"); SYNCHRONOUS REACH (2–6 arms, pounce or straight
  strike); PARACHUTING (pounce on top, spread the WEB, then feel underneath); SNAPTRAP (two groups of 2–4 arms elongate
  synchronously in two directions either side of — often also above — the prey).
- CRABS -> synchronous, cat-like POUNCE led by arm 2; SHRIMP -> sequential: arm 2 leads, then neighbours 1 and 3 secure it,
  with a characteristic SWAY.
- The SECOND arm from the midline (R2/L2) is the dominant strike arm for every prey type; arm on the SAME SIDE as the viewing
  eye (monocular drive). Adjacent arms recruited over other available arms.

### 32. Sivitilli, Strong, Weertman, Ullmann, Smith, Gire 2023 — Mechanisms of octopus arm search behavior without visual feedback
bioRxiv 10.1101/2023.03.13.532148 (published Bioinspir Biomim 2023? not checked). **Read: FULL** (bioRxiv, via summarising fetch).
- Occluded food search with an arm along a wall: a DISTALLY PROPAGATING WAVE OF SUCKER RECRUITMENT, initiated and continually
  regenerated, led by the PROXIMAL arm; a distal wave of deceleration, curvature and movement toward the wall.
- After distal suckers find the reward, free proximal suckers are "REELED IN" toward it (distal-to-proximal) — the arm
  gathers itself onto the find.
- Game implication: a "grope/search" primitive distinct from reach: sliding contact with sucker recruitment travelling to
  the tip; on find, the arm reels proximally toward the find point.

### 8b. Addendum to Levy et al. 2015 (crawling) from secondary sources (search snippets of the paper page, its Curr Biol
dispatch "Octopus Movement: Push Right, Go Left", DOI pii S0960-9822(15)00268-7, and a later review) — SECONDARY, verify:
- Stereotyped pushing arm sequence: (1) SHORTENING of a proximal arm segment, (2) a group of suckers ANCHORS to the substrate,
  (3) ELONGATION of the proximal segment generates thrust. During elongation the anchored suckers and the arm distal to them
  have ZERO velocity (the distal arm is a passive anchor; all motion is in the proximal segment).
- Between ONE and THREE arms push at each moment (review statement). Pushing directions are fixed per arm; crawl direction =
  unweighted vector sum. Fourier analysis: no periodicity in arm recruitment.
- Compare Katija 2025 (#26): in the deep-sea wild forward crawl ~3 ± 1 arms in contact, cyclical ventral->dorsal use, and the
  same shorten-then-elongate proximal segment with a bend at the "foot". The two are consistent on the per-arm primitive and
  disagree on whether inter-arm timing is rhythmic (lab glass wall vs rough wild terrain).

### 33. Reviews read as abstracts only
- Hochner 2012, "An embodied view of octopus neurobiology", Curr Biol 22:R887–R892, DOI 10.1016/j.cub.2012.09.001. ABSTRACT ONLY (not even fetched; known review).
- Hochner, Zullo, Shomrat, Levy, Nesher 2023, "Embodied mechanisms of motor control in the octopus", Curr Biol 33:R1119–R1125,
  DOI 10.1016/j.cub.2023.09.008. ABSTRACT ONLY: higher motor centres control and coordinate the arms WITHOUT body-part maps;
  control embodied in morphology from the muscle level up. The best single up-to-date review for Erik to read.
- Gutnick, Byrne, Hochner, Kuba 2011, "Octopus vulgaris uses visual information to determine the location of its arm", Curr Biol
  21:460–462, DOI 10.1016/j.cub.2011.01.052 — ABSTRACT ONLY: octopuses CAN guide a single arm through a 3-choice maze to a
  visually marked goal (vision tells the brain where the arm is).
- Gutnick, Zullo, Hochner, Kuba 2020, Curr Biol 30:4322, DOI 10.1016/j.cub.2020.08.037 — ABSTRACT ONLY: the CNS also uses arm
  proprioceptive/tactile info in a single-arm maze; arms autonomous but subject to central control.
- Nesher, Levy, Grasso, Hochner 2014, Curr Biol 24:1271, DOI 10.1016/j.cub.2014.04.024 — ABSTRACT ONLY: a chemical in octopus
  SKIN inhibits sucker attachment, so arms do not stick to each other or the body (amputated arms never grab octopus skin;
  the brain can override). -> arms crossing/touching each other must NOT adhere.
- Zullo, Sumbre, Agnisola, Flash, Hochner 2009, Curr Biol 19:1632, DOI 10.1016/j.cub.2009.07.067 — ABSTRACT not fetched;
  higher motor centres (basal lobes) are NOT somatotopic; stimulation evokes whole-body/multi-arm movements.
- Mather 2021/2023 commentaries and Hanlon & Messenger "Cephalopod Behaviour" (2nd ed. 2018, Cambridge UP) — not read;
  Hanlon & Messenger is the standard ethogram book (locomotion, web-over/pounce, "moving rock", jetting) Erik should have on hand.

## Gaps I could not close (numbers not found in anything I could read)
- Octopus JET cycle timing (inflate/contract durations, strokes per escape, frequency). Only squid data (oval squid peak
  contraction 3.8–8.6 mantle circumferences/s; thrust peak ~40 ms after jet onset in hatchlings) and O. vulgaris max speed
  (~100 cm/s, Wells 1990) and the arm-swim recovery/power ratio ~2.5 (FORTH group). Trueman & Packard 1968 (JEB 49:495,
  DOI 10.1242/jeb.49.3.495) has mantle pressure traces for O. vulgaris and Eledone (peak ~170 cm H2O in Octopus; Octopus jet
  thrust never above half its body weight, unlike squid/Sepia/Eledone) — fetch it for the cycle timing.
- Levy 2015 per-push duration and elongation strain; Sumbre 2005/2006 joint-angle profiles and fetch duration; Yekutieli
  wave travel times. All in paywalled full texts.
- The 12-action wild ethogram definitions (Bennice 2025 Fig. 2) — in an image.

---

# SYNTHESIS

Verdict on the design hypothesis: SUPPORTED, with three amendments.
(a) The brain-issues-goals / arm-runs-primitive split is the field's own model (Sumbre 2001; Flash & Hochner 2005; Levy &
Hochner 2017; Hochner et al. 2023): the higher centres choose WHICH arm, WHICH primitive, its DIRECTION and its AMPLITUDE/speed,
and the arm's own circuitry generates the shape.
(b) Each "primitive" in the list is itself a composition of a few lower primitives — bend propagation, elongation/shortening,
stiffening, torsion, and sucker-group attach/release — mixed in varying proportions (Hanassy 2015; Kennedy 2020). Build the stack
as goal -> primitive -> {stiffness front, elongation field, local bend/joint, twist, sucker groups} -> pose, not goal -> pose curves.
(c) Crawling is not purely gait-free: the lab says no rhythm (Levy 2015); the wild deep-sea data show a loose ventral->dorsal
cycle with ~3 feet down (Katija 2025). Treat inter-arm timing as a recruitment policy with an optional soft cycle bias.

## 1. The primitive vocabulary (with numbers)

**REACH (extension by bend propagation, plus elongation).** A bend forms near the arm base, often by first TWISTING the arm,
the distal part briefly pulled back (a "pre-extension" wind-up; Gutfreund 1996, Zelman 2013), and travels base->tip. The bend
is always curved DORSALLY, so the sucker face points the way the arm is going. The arm proximal to the bend is straight and stiff
(a stiffening front driven by a wave of muscle activation that leads the bend by 50–300 ms); the arm distal to it trails
passively. The bend point moves in ONE plane along a slightly curved path from the body centre to the target. Bend speed peaks
at 5–50 cm/s (O. vulgaris, 0.2–0.7 kg); the bend travels ~26–33 cm. The speed profile is one invariant shape (variable start,
monotonic acceleration to a LATE peak — ~0.55 ± 0.32 s after the bend passes a proximal point — then deceleration until the bend
dissolves at the tip), scaled by peak speed and distance. Peak speed is set at LAUNCH by activation amplitude (feed-forward).
Derived, not measured: ~30 cm at a mean of perhaps 15–25 cm/s gives a reach of ~1–2 s. The dynamic model needs only two numbers,
wave travel time and amplitude, and amplitude can rise (stiffer arm) without changing the kinematics (Yekutieli 2005). The
proximal segment ELONGATES during the reach: mean strain 0.57 ± 0.41 (range −0.12 to 1.8), 0.64 ± 0.28 in steady conditions
(Hanassy 2015) — the arm behind the bend gets ~1.6x longer and visibly thinner (constant volume), its sucker spacing opening.
A second reach type starts with a bend already distal and is mostly straightening and elongation of the part proximal to it.
The bend can launch from any station (Zullo 2019; at a hole, Richter 2015; from a sucker anchor, Grasso 2008). First contact
with a target is made with the MIDDLE THIRD of the arm (Byrne 2006b; Grasso 2008), the arm rotating to present the sucker face
(Kennedy 2020). Two arms reaching in the same context have near-identical speed profiles; a second arm follows 0.4–1.3 s after
the first. Broadside drag is ~50x tangential drag, which is why only the bend is ever broadside to the water.

**FETCH (pseudo-articulated transport to the mouth).** After suckers near the tip grip the object, two activation waves start
toward each other — one from the grasp site proximally, one from the base distally — and the medial joint (the "elbow") forms
where they meet. Result: three stiff segments and three pseudo-joints (base "shoulder", elbow, grasp "wrist"), the PROXIMAL AND
MEDIAL SEGMENTS OF EQUAL LENGTH, so the "hand" (the distal segment's group of suckers) lands on the mouth at the centre of the
circle of arm bases. The motion is mainly ROTATION AT THE ELBOW, and its invariants are in joint-angle space (Sumbre 2005, 2006;
Levy & Hochner 2017). Joints are rebuilt for each fetch according to where the object is held: grasp station s_g -> elbow at
~s_g/2. Durations and joint-angle profiles are in the paywalled papers. If articulation is blocked the arm simply PULLS IN by
shortening (Richter 2015; Kennedy 2020) — a separate "pull-in" primitive. In the wild, fetch uses only 2 of the 12 arm actions
(Bennice 2025): a clean, stereotyped motion.

**CRAWL STEP (push by proximal elongation).** Per arm: shorten the proximal segment -> plant a MID-ARM sucker group (the "foot")
-> hold a tight bend (a knee-like joint) at the foot -> push the body by ELONGATING the proximal segment (strain concentrated in
the third of the arm just proximal to the foot; anchored suckers and everything distal have ZERO velocity) -> release when the
proximal segment reaches maximum strain (suckers lose purchase) -> contract the proximal segment and swing the arm forward UNDER
the body for the next plant (Levy 2015; Katija 2025). The distal third barely changes shape during the step. It is a push, not a
pull. Speeds: A. aculeatus crawls at 7.3 ± 2.1 cm/s, 0.62 BL/s, max 21 cm/s; O. vulgaris averages ~9 cm/s (Huffard 2006;
Wells via Huffard). Absolute speed does not scale with body size. Climbing a wall uses another primitive: project the arm,
attach distal suckers, SHORTEN to pull the body up (probe with 12–15 suckers, then hold on 2–3; ~8 s per pull; Grasso 2008).

**BIPEDAL WALK / ROLL (a special-case gait).** Arm pair IV only, alternating, each arm ROLLING along its sucker edge like a tread;
the animal moves backward or obliquely backward; duty factor > 0.5; 13 ± 12 cm/s, 1.34 BL/s (max 19 cm/s, 2.25 BL/s); the other
six arms held up/coiled or tucked as camouflage (algae, coconut). A third or fourth arm touches down briefly at low speed. In
the wild, roll is mostly done by posterior arms too (71 %; Bennice 2025). A flavour behaviour, not the default gait.

**JET (escape) and SWIM.** Jet: funnel aimed forward, so the animal moves BACKWARD, mantle first; body fully elongated, arms
STRAIGHT and held TOGETHER trailing (or in a V, or spread laterally in large species for lift); skin smooth and pale. Speed
45 ± 19 cm/s mean, 70 cm/s max (A. aculeatus); ~100 cm/s for O. vulgaris. Range only a few metres: the systemic heart stops
during jetting and the oxygen debt it can carry is small (Wells 1987). So a jet is a short burst, after which the animal
settles and crawls. Swim (forward, head and arm crown first, funnel bent back): 10.8 ± 6.4 cm/s, 1.27 BL/s; the flat posture
(dorsoventrally compressed, arms forward and to the side) is 23–59 % faster than head-raised. "Medusoid" swimming /
arm-swimming: arms and web open SLOWLY and close FAST like an umbrella, recovery:power duration ~2.5:1 (FORTH group, secondary);
symmetric harmonic flapping produces no thrust. No octopus-specific jet cycle timing was found: the gap is real. Derive it from
footage (see section 6) or from Trueman & Packard 1968.

**Other primitives the literature names (worth having):**
- GROPE/SEARCH: a distally propagating wave of sucker recruitment along a surface, proximal arm leading; on a find, proximal
  suckers "reel in" toward it (Sivitilli 2023).
- STRIKE/SLAP: a stiffened arm ROTATED about its base plus TRANSLATION, combined serially or in parallel across arms (Alupay 2023).
- Prey strikes: REACH (1–3 arms, sequential); WAVING (torsional sway, no bend); SYNCHRONOUS POUNCE (2–6 arms); PARACHUTE
  (pounce, spread the web over the prey, feel underneath); SNAPTRAP (two groups of 2–4 arms elongating on either side) (Bidel 2022).
- ARM WALK: the arm translates sideways across a surface with the two sucker rows attaching in anti-phase, 6–15 suckers per
  group, up to ~8 cycles (Grasso 2008).
- The wild 12-action vocabulary: reach 19 %, raise 18 %, lower 17 %, tuck 14 %, curl 10 %, then parachute, roll, grasp, stilt,
  tiptoe, push (Bennice 2025).
- Underlying deformations: bend 67–70 %, elongate 16–22 %, shorten 6–8 %, torsion 2–9 %; the proximal third mostly
  elongates/shortens, the distal third mostly bends (Kennedy 2020; Bennice 2025).

## 2. How arms coordinate in crawling and swimming
- Crawl direction = the vector sum of the fixed pushing directions of the arms currently pushing. The body can crawl in ANY
  direction regardless of where it faces, and facing is controlled independently (Levy 2015). The controller picks, moment to
  moment, the arms whose pushing direction helps; active arms push with about equal force.
- How many: 1–3 pushing at a time (lab; review statement); 3 ± 1 in contact (range 2–6) in the wild, with both sides in contact
  73 % of the time (Katija 2025). Huffard's definition of a crawl: more than two arms, each touching at several points.
- Which: arm pair 1 (dorsal/front) PROBES ahead and is rarely weight-bearing; pairs 2–4 bear weight; ventral arms have shorter
  cycles and higher duty factors, dorsal arms longer and more variable cycles (Katija 2025). Front arms do 64–75 % of exploring;
  back arms stilt and roll (Byrne 2006a; Bennice 2025; Gendreau 2025).
- Timing: no fixed gait. FFT shows no rhythm in lab crawling (Levy 2015; Levy & Hochner 2017); in the wild a loose cyclical
  ventral->dorsal recruitment appears (Katija 2025). Recommendation: recruit by utility with a weak preference for the next arm in
  a ventral->dorsal sweep, never a lockstep gait.
- The head stays LEVEL through all locomotion (eye axis horizontal), achieved by adjusting the length and stiffness of the
  contact arms (Levy & Hochner 2017). Body height is set by the arms, not by a hip.
- Swimming and jetting: arms are NOT individually controlled. They are one bundle (trailing, together) or one umbrella
  (medusoid), coordinated with the mantle cycle.

## 3. Object holding and transport
- The "hand" is a GROUP OF SUCKERS on the distal segment (fetch); a power grasp is many suckers on one surface. Octopuses rarely
  wrap an arm around an object; wrapping looks like momentum after a collision (Grasso 2008; Levy & Hochner 2017).
- First contact and the preferred working zone are the MIDDLE THIRD of the arm, even where force would favour the big proximal
  suckers (Byrne 2006b; Grasso 2008). More force -> more suckers (10 -> 16 for a +28 % load).
- Only a minority of suckers are sealed at any time (18 ± 14 %, max 39 %), never all. Free suckers explore by extending and
  rotating with only the rim touching (Grasso 2008). Suckers orient toward attractive stimuli and away from aversive ones,
  recruit their neighbours, and pick one of two stimuli (Gribkova 2025). Extrinsic sucker muscles can rotate the rim, and the
  held object, about the arm axis (Grasso 2008).
- Multi-arm manipulation recruits ADJACENT arms (A+1 most likely), typically 3–5 arms at once, never all 8 (Gendreau 2025;
  Bidel 2022; Byrne 2006b).
- Lifting or moving a heavy object: drape the WEB over it, mouth over it, arms down its sides to the floor, and push it up by
  lengthening the arm between suckers on the object and suckers on the floor (Grasso 2008).
- Transport while moving: the object goes to the mouth/web by fetch or pull-in; carrying during locomotion uses the arms not
  needed for pushing (bipedal walkers hold six arms up as camouflage). I found no quantitative paper on sucker-to-sucker
  "conveyor" passing along an arm; the "reel-in" of Sivitilli 2023 is the closest. Treat conveyor passing as plausible but unmeasured.
- Arms never stick to their own or each other's skin: a skin chemical inhibits the sucker reflex (Nesher 2014).

## 4. What a game animation would most obviously get wrong
1. Continuous sine waves running down every arm ("tentacle wiggle"). Real arms are mostly held still, curled or tucked, and
   move in discrete, stereotyped events. Symmetric harmonic flapping is even hydrodynamically useless.
2. Reaching by curling the whole arm toward the target (IK-style uniform bend). Real: ONE travelling bend, a straight stiff arm
   behind it and a limp arm ahead of it, in one plane, suckers facing forward (dorsal bend), with a short twist wind-up first.
3. Fixed arm length. The real proximal arm stretches ~1.5–1.8x in reaches and crawl pushes and gets THINNER (constant volume),
   with sucker spacing opening; shortening thickens it.
4. Spider/insect gait tables: all arms stepping in rhythm and pulling with the tips. Real: 1–3 (lab) / ~3 (wild) arms down,
   planted with the MIDDLE of the arm and PUSHING by proximal elongation, while the distal part sits still; front arms probe.
5. Head bobbing and tilting with each step. Real: the head is held level.
6. Swimming head-first with arms paddling everywhere. The real escape is a JET, mantle first, arms straight and bundled behind,
   pale and smooth, a few metres at most, then back to crawling. Medusoid strokes are slow-open, fast-close.
7. Grabbing by coiling the tip around things. Real: sucker patches flat on the surface, mid-arm, few suckers sealed; fetch
   becomes three stiff segments with an elbow at the halfway point.
8. All suckers stuck down, or arms sticking to each other. Real: ~1 in 5 sealed; arms slide over each other freely.
9. All eight arms equally busy. Real: front pairs explore (the 2nd arm leads strikes), back pairs support; a goal recruits the
   nearest arm plus its neighbours.
10. Constant-speed motions. Real reaches have the invariant accelerate-late, bend-dissolves-at-tip profile.

## 5. Papers Erik must fetch (full text not readable to me)
Highest value first:
1. Levy, Flash, Hochner 2015, Curr Biol 25:1195 — DOI 10.1016/j.cub.2015.02.064 (crawl numbers, SI movies).
2. Sumbre, Fiorito, Flash, Hochner 2006, Curr Biol 16:767 — DOI 10.1016/j.cub.2006.02.069 (fetch joints and waves; the
   open-archive PDF at cell.com/article/S0960982206012747/pdf blocked my fetcher).
3. Sumbre, Fiorito, Flash, Hochner 2005, Nature 433:595 — DOI 10.1038/433595a (fetch; supp. video).
4. Sumbre, Gutfreund, Fiorito, Flash, Hochner 2001, Science 293:1845 — DOI 10.1126/science.1060976 (peripheral reach program).
5. Hanassy, Botvinnik, Flash, Hochner 2015, Bioinspir Biomim 10:035001 — DOI 10.1088/1748-3190/10/3/035001 (elongation).
6. Yekutieli et al. 2005, parts I and II, J Neurophysiol 94:1443 and 1459 — DOIs 10.1152/jn.00684.2004 and 10.1152/jn.00685.2004
   (dynamic model: the closest thing to a ready-made arm physics model for the game).
7. Hochner, Zullo, Shomrat, Levy, Nesher 2023, Curr Biol 33:R1119 — DOI 10.1016/j.cub.2023.09.008 (current review).
8. Bidel, Bennett, Wardill 2022, Curr Biol 32:4727 — DOI 10.1016/j.cub.2022.08.080 (prey-capture strike types; videos).
9. Huffard, Boneka, Full 2005, Science 307:1927 — DOI 10.1126/science.1109616, plus its Supporting Online Material (walking kinematics).
10. Trueman & Packard 1968, J Exp Biol 49:495 — DOI 10.1242/jeb.49.3.495 (mantle pressure and jet timing, O. vulgaris).
11. Wells et al. 1987, J Exp Biol 131:175 — DOI 10.1242/jeb.131.1.175 (jetting cost, cardiac arrest); Wells 1990,
    Can J Zool 68:815 (jet propulsion; DOI not verified).
12. Mather 1998, J Comp Psychol 112:306 — DOI 10.1037/0735-7036.112.3.306 (arm-use taxonomy); Mather & Alupay 2016 —
    DOI 10.1037/com0000025 (ethogram).
13. Alupay, Mather, Iskarous 2023, Mar Biol 170:99 — DOI 10.1007/s00227-023-04243-y (arm-slap primitives).
14. Richter, Hochner, Kuba 2015, J Exp Biol 218:1069 — DOI 10.1242/jeb.115915.
15. Sfakiotakis, Kazakidi, Tsakiris 2013, ICRA, "Octopus-inspired eight-arm robotic swimming by sculling movements"
    (projects.ics.forth.gr/cvrl/octopus/publications/ICRA13_2200_MS.pdf), for its arm-swimming biology section.
16. Kier & Smith 1985, Zool J Linn Soc 83:307 (muscular hydrostats; DOI not verified); Kier & Stella 2007, J Morphol 268:831 —
    DOI 10.1002/jmor.10548.
17. Byrne et al. 2006a and 2006b — DOIs 10.1037/0735-7036.120.3.198 and 10.1016/j.bbr.2006.04.026.
Already read in full (open access, no need to fetch): Gutfreund 1996, Gutfreund 1998, Zelman 2013, Zullo 2019, Levy & Hochner
2017, Kennedy 2020, Bennice 2025 (except its Fig. 2 image), Gendreau 2025, Gribkova 2025, Katija 2025, Huffard 2006,
Grasso 2008, Sivitilli 2023, Weidig 2024.

## 6. Reference footage
Lab and paper supplements (the most honest footage: known behaviour, known species):
- Katija et al. 2025, Nature, Supplementary Videos 1–2: deep-sea Muusoctopus crawling forward over rock, with 3D arm tracking
  (open access, nature.com/articles/s41586-025-09379-z). MBARI's YouTube channel carries Octopus Garden / EyeRIS footage.
- Levy et al. 2015, Curr Biol, SI movies; EurekAlert hosts two clips, "Crawling Octopus 1 of 2 / 2 of 2"
  (eurekalert.org/multimedia/650114 and /650113).
- Huffard 2006, JEB, supplementary Movies 1–3 (medusoid jetting, bipedal walking; jeb.biologists.org/cgi/content/full/209/19/3697/DC1)
  and the Huffard et al. 2005 Science SOM movies (coconut octopus and Abdopus walking).
- Sumbre et al. 2005, Nature, supplementary video (fetching with pseudo-joints).
- Kennedy et al. 2020, Sci Rep, Supplementary Videos 1–2 (bends, torsion, elongation; reaching, fetching, pull-in).
- Bidel et al. 2022, Curr Biol, videos plus the Dryad dataset (628 prey strikes: reach, pounce, parachute, snaptrap).
- Gendreau et al. 2025, Biol Open (arm recruitment into a hidden-item dome).
Documentaries and channels (titles known to me, NOT re-verified in this session):
- "My Octopus Teacher" (Netflix, 2020, Craig Foster): wild O. vulgaris crawling, jetting, hunting, hiding under shells,
  escaping a shark — the best long naturalistic sequences.
- "Secrets of the Octopus" (National Geographic / Disney+, 2024): hunting alongside fish, coconut octopus, crawling.
- PBS Nature, "Octopus: Making Contact" (2019).
- BBC "Blue Planet II", episode "Green Seas" (2017): an octopus armouring itself with shells against a pyjama shark.
- YouTube: MBARI (Monterey Bay Aquarium Research Institute); Nautilus Live / Ocean Exploration Trust (EV Nautilus ROV dives,
  deep-sea octopus crawling and swimming); NOAA Ocean Exploration; Monterey Bay Aquarium; Marine Biological Laboratory
  (Hanlon lab clips of mimic octopus and Abdopus walking).
