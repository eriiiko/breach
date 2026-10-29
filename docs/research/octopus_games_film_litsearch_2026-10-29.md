# Octopus strand (c): games, film and graphics prior art for living tentacles — digest

Date: 2026-10-29 · Issue context: Breach #105 (octopus animation study), strand (c).
Companion strands: (a) motor-control biology, (b) soft-arm simulation.
Question: what did GAMES, FILM and GRAPHICS actually do to make tentacles and multi-limb
creatures look alive, and what can a top-down, procedurally posed 8 x 17-bone octopus copy?

Honesty convention per source: **Read:** says exactly what was read (full paper / full
article / full transcript / abstract only / secondary summary only).

## Intended source list (as planned before searching)

1. Games
   - Rain World (Joar Jakobsson / Videocult) — GDC/interviews/devlog on procedural creatures
   - Spore (Hecker et al., SIGGRAPH 2008, motion retargeting to varied morphologies)
   - Overgrowth (David Rosen, GDC 2014, "An Indie Approach to Procedural Animation")
   - Carrion (Phobia Game Studio) tentacle monster tech
   - Inside / Limbo (Playdead) — the Huddle; procedural animation talks
   - Half-Life 2 / Alyx barnacle and tentacles; Dead Space
   - Subnautica sea creatures; Abzu (fish/boids)
   - Octodad (what NOT to copy)
   - Sea of Thieves kraken; Hades/God of War hydra; Elden Ring; Monster Hunter; Horizon snake machines
2. Film / VFX
   - Davy Jones (Pirates of the Caribbean, ILM) — beard tentacles rig + sim
   - Doctor Octopus (Spider-Man 2, Sony Imageworks)
   - Kraken (Clash of the Titans); Arrival heptapods
   - Tentacle-rig tutorials (spline IK + FK layered, wave deformers)
3. Graphics research
   - VIPER (Angles et al. 2019); position-based elastic rods; Jakobsen (Hitman) Verlet
   - Active ragdolls (Wolfire / GDC); motion matching for non-humanoids
   - Procedural spider / multi-leg gait write-ups; Unreal Control Rig; Houdini KineFX / Vellum
4. Craft
   - Hank (Finding Dory, Pixar) — SIGGRAPH talks on rigging/animating Hank
   - Ursula (Disney); animation principles applied to tentacles
5. Top-down presentation readability

---

## Source notes (appended as read)

### 1. Talbot, Piretti, Singleton, Hessler — "Designing an Interaction with an Octopus" (Pixar, SIGGRAPH 2016 Talks)
- DOI 10.1145/2897839.2927434 · free PDF: https://history.siggraph.org/wp-content/uploads/2022/09/2016-Talks-Talbot_Designing-an-Interaction-with-an-Octopus.pdf
- **Read:** full text (it is a 1.25-page talk abstract; nothing more exists in print).
- Hank's arms are NOT posed through a bone hierarchy: animators pose **"knot objects"** in 3D directly —
  control points on a specialised spline, borrowing the language of a 2D curve editor. Every knot can
  **slide along the arm**, be constrained to world/objects, and switch behaviours (constrainability,
  smoothing, shaping). Uniformity of knots = learnability.
- Key physical observation they designed around: **"Octopus arm deformation does not appear at fixed
  locations along its length. Rather, shape change needs to be able to float through the geometry."**
  This is the rig-side statement of the biology's travelling bend — a bend site is a moving parameter,
  not a joint.
- Level-of-detail hierarchy of curves: coarse curves for broad shape, finer curves for refinement;
  deformation weights mapped from knot to knot ("parameterization sites").
- **Arm-to-body continuity is a design requirement**: curves pass from each tentacle into the body so
  the arc of the arm continues into the mantle/web — "preventing the sense of a break or
  discontinuity". Director brief: "precise control and graceful, clean silhouettes", no "disjointed
  shapes or breaks in form" between arm and body.
- Layered deformation: curves → progressively denser proxy meshes → final topology; web and arm roots
  on coarse meshes, face on dense; correctives via **PSD (pose-space deformation) driven by the
  relative angle of the body to each arm**.
- No control counts given. For us: our 17 bones are the "final topology" level; the authored layer
  should be a handful of SLIDING knots per arm, not 17 FK angles; and the web/crown between arm roots
  needs an explicit continuity treatment (arm curvature carried into the mantle skin).

### 2. Kautzman, Wise, Yu, Karlsson, Hessler, Wong — "Finding Hank: Or How to Sim an Octopus" (Pixar, SIGGRAPH 2016 Talks)
- DOI 10.1145/2897839.2927458 · free PDF: https://history.siggraph.org/wp-content/uploads/2022/09/2016-Talks-Kautzman_Finding-Hank-Or-How-to-Sim-An-Octopus.pdf
- **Read:** full text (2 pages).
- Layering: animation rig is the truth; simulation ADDS flesh/skin secondary motion on top (PhysBAM
  FEM tet meshes). Separate meshes for body, suckers, mantle.
- **Mantle**: coarser tet mesh = fewer DOF = easier to art-direct. Where mantle collision was not
  visible they replaced sim with **a linear spring tracking the mantle centroid, applied on top of the
  animation** for secondary motion; the same springs drove Hank's "ears". → Game-cost copy: a damped
  spring on the mantle (and per-arm-root jiggle) layered over the procedural pose.
- **Suckers (~350 instances)**: a **pre-sim heuristic script** decides stick/release from *distance to
  surface* and sucker *rotation from the animation*; sim then targets art-directed compress/release
  shapes. All suckers share identical topology so dynamics are size-independent. → For us: sucker
  "stiction" is a rule (distance + orientation), not physics.
- **Skin sliding** over the animated surface, but sliding forbidden in creases/boundaries (curvature
  test).
- The rig's control modes per arm control: world-space translate, **FK bend angles as if a chain of
  bones**, or constrained to any object/other control. I.e. a hybrid: positional knots + FK + constraints.
- Pre-roll was generated **procedurally**: rig set to a canonical state, arms unfolded sequentially,
  then the rig's "automatic posing behaviours" smoothly engaged. → Even Pixar's rig had automatic
  arm-shape management that animators relied on.

### 3. Press interviews on Hank's animation (supervising animator Michael Stocker; character art director Jeremie Talbot)
- KSL/AP (2016): https://www.ksl.com/article/40251733/ · Bustle (2016): https://www.bustle.com/articles/167194 ·
  thatsitla (2016) · Animation Magazine "A Tentacle Spectacle" (403, not read; content via search snippet).
- **Read:** KSL, Bustle, thatsitla in full (short press pieces); Animation Magazine only via search-result summary.
- **"Early tests had every tentacle moving at once — it was too hard to watch." They cut to two or three
  arms moving at a time to keep the focus on Hank's eyes.** (Stocker, KSL/AP.) This is the single most
  transferable craft rule for us: most arms rest, few act.
- **Lead from the "elbow", not the tip**: animators used a "grab cylinder" they called the elbow and
  "had to lead with it rather than the end of the tentacle to get an octopus feel" (Stocker, via
  Animation Magazine snippet). Matches the biology: a reach is a bend travelling outward, the tip follows.
- Sketch-to-pose: animators **drew a tentacle curve and snapped the model to the drawing — "60 percent
  of the way"**, then polished. Authoring a tentacle = authoring a 2D curve.
- A webbing-depth control between arms; an engineer spent ~6 months on the new tentacle curve type
  (squash, stretch, slide). Rig took ~18 months. One shot (Hank sliding into an aquarium) took months.
- "Any contact, Hank moving across any surface becomes really hard" — contact is where cost goes.
- Earlier Pixar tentacles (Monsters University) were "simplified versions"; every Hank tentacle got its
  own interface.

### 4. Davy Jones' beard (ILM, Pirates of the Caribbean 2 & 3)
- Criswell, Derlich, Hatch, "Davy Jones' beard: rigid tentacle simulation", SIGGRAPH 2006 Sketches,
  DOI 10.1145/1179849.1179995 (ACM DL; no free PDF found). Cites Weinstein/Teran/Fedkiw 2005
  (articulated rigid bodies with contact) and Guendelman et al. 2003.
- CGW "Yo Ho Ho" (July 2006): https://www.cgw.com/Publications/CGW/2006/Volume-29-Issue-7-July-2006-/Yo-Ho-Ho-.aspx ·
  CGW "All Hands on Deck" (May 2007): https://www.cgw.com/Publications/CGW/2007/Volume-30-Issue-5-May-2007-/All-Hands-on-Deck.aspx
- **Read:** sketch abstract only (one sentence); both CGW articles in full (via fetch summaries).
- **46 tentacles, each a chain of rigid cylinders ("stacked soup cans"), up to ~40 segments.** Earlier
  ILM rigid-body chains could only spin segments; for Davy they put **motors at the joints** →
  articulated rigid-body dynamics where the motors are driven by **expressions or keyframes**, and left
  alone "try to move on their own".
- **Curling = sine waves on the motors**, with TDs choosing which motors run forward/backward so a
  tentacle curls like an octopus arm while moving in 3D. → a phase-offset sinusoid along the chain IS
  the classic cheap "alive tentacle" drive; ILM shipped it at hero quality.
- A **"super controller"** (lead creature TD Karin Derlich) exposed a few global dials — e.g. motor
  speed — over hundreds of joints. **Mood as a parameter: "when Davy gets angry, the tentacles get
  angry."** → Our equivalent: an arousal/agitation scalar that scales amplitude, frequency and
  stiffness across all 8 arms.
- **"Sticktion"**: tentacles stick slightly to Davy's own body as they swish ("wet spaghetti on a
  leather jacket") — contact friction is a look, and a big one.
- Animators keyframed the sneers (facial tentacles carry the actor's expression); otherwise the beard
  was procedural/simulated — **~95 % of the time simulation moved the tentacles** (CGW 2007), and the
  sim was smooth enough that the director "could direct the tentacles to create a performance".
- Tet-mesh flesh sim ran on top for jiggle/volume preservation in some shots; suction cups vibrate
  when pushed by collision objects.
- Lesson: at hero quality, a **motorized joint chain** (target-tracking joints with a procedural
  sine drive + a few global dials + stiction) was enough for ~46 background-ish tentacles; full
  animator control was reserved for the few "acting" tentacles. That is the game architecture.

### 5. Doctor Octopus (Spider-Man 2, Sony Imageworks + Edge FX, 2004)
- CGW "Another Big Leap" (July 2004): https://www.cgw.com/Publications/CGW/2004/Volume-27-Issue-7-July-2004-/Another-Big-Leap.aspx ;
  Slashfilm on Molina naming the arms: https://www.slashfilm.com/738482/ ; moviemistakes trivia page.
- **Read:** CGW article in full (fetch summary); naming/personality only via search-result summaries
  (trivia-grade sources — treat the "personality" claim as anecdote).
- CG arm = "vertebrae strung on a cord, like beads on a string", built "like a big Lego set";
  contracting/expanding cables held them and made them **vibrate** (a tremor layer).
- Practical: 4 foam tentacles, ~2.4 m each, 16 puppeteers on set.
- **"It was like animating five characters"** (animation director Anthony LaMolinara) — each arm is an
  agent. Molina named them (Harry, Larry, Flo, Moe): the lower pair = heavy load-bearing, Flo = the
  delicate manipulator; puppeteers reportedly gave each a distinct personality.
- Animators sold **power and weight** — how much the arms can lift — in the hospital and climbing shots.
- → For us: **role specialisation per arm** (biology also has it: front arms explore, rear arms
  walk/anchor) plus small per-arm "temperament" offsets is a cheap, proven way to stop eight copies of
  one controller from reading as a machine.

### 6. Rain World (Joar Jakobsson / Videocult, 2017) — devlog + GDC 2016 Animation Bootcamp
- Devlog archive (TIGSource thread mirrored): https://candlesign.github.io/Rain-World-Devlog/Full%20devlog
- GDC 2016 Animation Bootcamp, Jakobsson & James Therrien, free video: https://www.youtube.com/watch?v=sVntwsrjNe4
  (via https://www.gamedeveloper.com/art/video-animating-i-rain-world-i-and-its-many-squishy-stretchy-creatures)
- Unity blog on The Watcher (2025): https://unity.com/blog/exploring-procedural-design-rain-world
- **Read:** devlog characters 0–200k, 300k–400k, 500k–700k of ~786k (via fetch-summaries; the
  Daddy-Long-Legs / tentacle-creature posts were NOT found in the parts read); the GDC talk itself NOT
  watched (no transcript retrievable); Unity blog in full.
- **Body = a few free Newtonian point-masses ("chunks") joined by distance constraints**; lizards 3
  chunks, slugcat 2, tails are chains of points. "Not animated by traditional means, but … a few freely
  moving pieces." Bodies got a "mushy, soft quality" he deliberately kept. Limbs/tail are drawn ON TOP
  of the physics chunks (cosmetic for the player; propulsive for lizards).
- **Limb stepping rule (lizard)**: a limb that falls too far behind the body enters **search mode**,
  finds a grasp point *ahead* on terrain (point-to-polygon distance for slopes), and **stays planted
  until the body leaves it behind**. **Propulsion scales with the number of gripping limbs**: 4 grips =
  fast, all in the air = "body barely moves"; a weak fallback force so it can't get stuck. The
  resulting hauling, uneven gait was intended ("creepy").
  → This is exactly the octopus crawl: plant, haul, release; locomotion speed EMERGES from contact
  count, which reads as effort and weight. Copyable 1:1 at game cost.
- **Animation reads AI state**: during path computation the lizard "paces back and forth" rather
  than freezing — "its speed will be a cue to what behaviour it's engaged in"; on hearing a sound it
  stops "as if listening", turns its head to the source, freezes briefly, then investigates. Head
  turns were too subtle at sprite size → he ADDED a particle cue. Head colour shows "has spotted you".
  → Rule: every AI state change needs a visible body tell, exaggerated until it reads at game scale.
- Per-creature "brain ghost" of other creatures (where it thinks you are) drives gaze/approach —
  attention is pointed at a *belief*, which is what makes it read intelligent.
- Engineering: lizards update graphics every 4th frame (cost); early per-limb vector drawing cost
  8–9 ms → swapped for distance-indexed sprite frames. Animation is "less procedural than it sounds":
  many special cases, chosen per moment by which force acts on which body part.
- Third-party note (search summary only): a dev who rebuilt Daddy-Long-Legs from the talk kept data
  flowing ONE way (AI → body → graphics) to avoid feedback loops.

### 7. Hecker, Raabe, Enslow, DeWeese, Maynard, van Prooijen — "Real-time Motion Retargeting to Highly Varied User-Created Morphologies" (Spore; ACM TOG 27(3) art. 27, SIGGRAPH 2008)
- DOI 10.1145/1360612.1360626 · free: https://chrishecker.com/Real-time_Motion_Retargeting_to_Highly_Varied_User-Created_Morphologies
  (copy read: Chalmers course mirror PDF).
- **Read:** full paper, all 11 pages.
- **Pipeline (§4)**: authored channels specialised onto the body → **pose goals** → + synthesised
  locomotion → **Particle IK solver** → **passive secondary "Jiggles"**. That is the same
  intent → shape → dynamics stack, shipped, at **0.2 ms/frame for a 25-body creature on a 1.7 GHz
  Pentium-M (35 % in IK)**. Creatures 20–80 bodies.
- **Particle IK = Jakobsen-style length constraints between 3-DOF particles**, nonlinear length
  correction, per-constraint per-endpoint "masses" as tuning knobs (no physical meaning), **soft
  constraints via a C1 piecewise response; limbs may stretch/compress (10 %/120 % spine projection
  bounds) "to give an empirically more organic feel"**. 5 outer iterations. They tried CCD, Jacobian
  and constrained-dynamics solvers and found them slower and *less tunable*; the simple solver gives
  "local control" for ad hoc fixes. → Strong prior-art endorsement of an XPBD/Verlet-class arm solver.
- **Aim preconditioner** (key trick): before iterating, rotate+scale the whole limb so its tip sits on
  the goal (minimal-twist rotation from rest-chord to goal-chord + a stretch along it), then let
  constraints restore lengths. "Simply running the constraint iterations from the rest pose leads to
  unnatural poses, as the goal error is distributed poorly along the limb." → For an octopus reach,
  initialise from a sensible *shape* (the travelling-bend profile), never from the last pose or a
  straight line, then project.
- **Spine splines**: particles only at IK branch points; interior bodies reconstructed from a fitted
  spline by frame transport (they needed **quintic Hermite** — cubics could not fit player spines with
  many inflections). → Our 17 bones can be reconstructed from a few solved knots + a spline with
  parallel-transport frames (exactly Hank's "knots" pattern, at game cost).
- **Anti-buckling**: detect fold-over by comparing neighbouring constraint angles, then smoothly
  blend toward the rest pose. **Path independence** is an explicit requirement — "highly redundant
  manipulators with path dependent solvers can eventually tie themselves in knots" — the direct
  warning for 17-bone arms. Path-independence brought singularities (hairy-ball) they pushed to
  unlikely poses.
- **Gaits**: legs grouped by length; groups' frequencies harmonised by small rational ratios; per foot
  **duty factor** + **step trigger** (phase offset); speed→gait-parameter maps authored for 1–6 feet,
  generated procedurally for ≥7. Footless characters get spine "pseudo-feet" and an **inch-worm gait**.
  Stated biggest challenge: "believable locomotion with **anticipation** in the face of discontinuous
  player and code inputs."
- **Variants**: an action authored once plays on "whichever grasper is best" (closest to target, not
  already holding) chosen by game code → our "which arm reaches" selection.
- **Lookat** with soft limits, layered under authored nods. **Jiggles must be passive** — the earlier
  "Wiggles" fed secondary motion back as IK goals and "negatively impacted the quality of the
  animators' work". → Rule: secondary dynamics never feed back into the intent layer.
- Limitations they name: no self-collision ("clearly a flaw"), no notion of volume (can't "hug").
  Testing: ~90 % pass rate over hundreds of creatures × ~1000 animations, via a validation grid.

### 8. David Rosen (Wolfire) — "An Indie Approach to Procedural Animation", GDC 2014 Animation Bootcamp
- GDC Vault (free video): https://gdcvault.com/play/1020049/Animation-Bootcamp-An-Indie-Approach ;
  YouTube https://youtu.be/LNidsMesxSE
- **Read:** NOT watched; no transcript retrievable (YouTube blocks captions to the fetch tool, Polycount
  thread 403). Content below is from secondary summaries (Game Developer, gameanim.com, Unity
  forum) plus my own prior knowledge of the talk, flagged as such.
- Summaries agree: very few keyframes (posters: ~13 keyframes for the whole character; cycles of 2–4
  poses), interpolated with spring-like, non-linear curves, with procedural layers on top; sections on
  refinement and ragdolls; trigonometry rather than heavy IK.
- *(Prior knowledge, unverified here)*: the run/walk phase is driven by distance travelled (a "wheel"
  rolling under the character) so feet never slide; physics-based secondary motion (springs on body
  parts, ears, weapon) and an active-ragdoll blend for hits.
- → For us: a tiny vocabulary of key shapes + good interpolation + springs + phase-locked-to-distance
  is the proven indie path to "fluid"; it argues our primitive library can be small.

### 9. Carrion (Phobia Game Studio / Devolver, 2020) — the tentacle-mass monster
- PlayStation Blog dev post (Sebastian Krośkiewicz): https://blog.playstation.com/?p=355387 ; Wikipedia (MonoGame engine).
- **Read:** PS Blog post in full (fetch summary); no technical postmortem found.
- Monster = many co-interacting physics bodies responding to the same input; "game-breaking physics
  bugs were more likely" because of it; fixes were dressed as mechanics/lore. **Tentacles are
  physically simulated once they detach from a surface** (i.e. grip = kinematic anchor, free =
  passive physics). Making the core movement "feel rock-solid and fun" was the bulk of the work, not
  writing it. Sound carried much of the organic read (whip = a USB cable cutting air).
- Reviewers' reading (not technical): it "sprouts new appendages as it needs them" — tentacles appear
  *where* the creature is about to grip, which sells intent.
- → For us: anchored-or-free per arm, with contact the default and free-swinging the exception;
  budget for stability hacks; audio is part of the read.

### 10. Thomas Jakobsen — "Advanced Character Physics" (IO Interactive, GDC 2001; Hitman: Codename 47)
- Copy read: https://www.cs.cmu.edu/afs/cs/academic/class/15462-s13/www/lec_slides/Jakobsen.pdf (19 pp.)
- **Read:** full paper. (Strand (b) covers the numerics; here only the game-look lessons.)
- **"Believability … the programmer can cheat as much as he wants if the player still feels
  immersed"**; stability is part of believability (no vibration at rest, no blow-up).
- Verlet + projection + relaxation; **1–10 iterations per object type in Hitman**; for cloth and
  plants **one iteration was enough, and "the low number gave the plants exactly the right amount of
  bending behaviour"** — under-converged constraints read as organic softness; 3–4 iterations for
  bodies, "not completely rigid … mostly a nice feature".
- Hitman corpses = stick figures (particles + sticks); limb twist about its axis is NOT simulated — the
  skeletal layer orients limbs for rendering. → Same split for us: solve a particle chain, derive bone
  frames (twist from parallel transport / rules) at render.
- **Motion control by moving particles**: inherit velocity from animation by feeding two frames of
  positions; IK by pinning a particle (invmass 0) inside the loop. Soft constraints = repair only a
  fraction of the error per frame. Friction by trimming tangential velocity ∝ penetration depth.
  Inaccuracy is largest under heavy motion, "which fast or complex motion masks for the eye".

### 11. ABZÛ (Giant Squid, 2016) — vertex-shader swimming (Matt Nava, "Creating the Art of ABZÛ", GDC 2017)
- GDC Vault overview: https://gdcvault.com/play/1024409/ (talk not watched). Technique write-up modelled
  on it: Godot docs "Animating thousands of fish" https://docs.godotengine.org/en/3.2/tutorials/3d/vertex_animation/animating_thousands_of_fish.html ;
  MIT reimplementation https://www.github.com/frankiezafe/Fish-shader
- **Read:** Godot doc in full; the talk itself NOT watched (secondary only).
- No bones: four cosine motions in the vertex shader in model space — side-to-side, pivot, a
  **travelling wave `cos(t + body_coord)`** along the spine, and a twist (roll about the spine) with the
  same phase. **Masked to the rear half (`smoothstep` along the body)** because fish swim with the back
  half; twist is a `mix` (not a scale) so the body doesn't shrink. **Per-instance phase offset + 50–150 %
  speed** desynchronises the school.
- → Relevance: a cheap render-only "breathing" layer (sub-bone undulation, arm-tip curl, web ripple)
  can live in the vertex shader on top of the 17-bone pose, never touching the sim. And per-arm phase
  and rate offsets are the minimum anti-synchrony measure.

### 12. Sea of Thieves Kraken (Rare, 2018) — a negative example
- Windows Central interview with the design director: https://www.windowscentral.com/sea-of-thieves-kraken ;
  criticism: https://gamerant.com/sea-of-thieves-kraken-design-disappointing/
- **Read:** search-result summaries of both (no technical write-up exists that I could find).
- Design: 8 independent tentacles, coordinated by one central Kraken AI; wrap / rock / punch the ship.
- Reception: **"the tentacles connect to absolutely nothing — no body, no eyes"**, and fighting it is
  "unloading cannons into some mostly static tentacles". → Two failure modes to avoid: (1) arms without
  a visible, reacting body/eye read as props, not as one animal; (2) arms that hold still under fire
  read as dead. Every arm motion should visibly originate in, and be answered by, the mantle/eyes.

### 13. Octodad: Dadliest Catch (Young Horses, 2014) — what NOT to copy
- GDC Next 2014 "There and Dad Again" (Murphy, Tibitoski): https://gdcvault.com/play/1021471/ ; Notacon 9
  (2012) "Octodad: Building a Better Tentacle Ragdoll" (Devon Scott-Tunkin) listing only.
- **Read:** listings and search summaries only; no technical text found. The analysis below is MINE,
  from the game's well-known control scheme, not from a source.
- The player drives each limb's TIP directly (mouse/stick per limb) and the rest of the limb is a
  passive physics chain dragged behind it. The comedy comes precisely from that: (1) **the tip leads
  and the body follows** — the reverse of Hank's "lead with the elbow"; (2) passive dynamics dominate
  intent, so nothing anticipates and nothing settles; (3) every limb is equally active (no rest, no
  focus). Read as: clumsy, drunk, not-in-control.
- → Our octopus must invert all three: proximal-first motion (travelling bend from the base),
  intent-driven shape with dynamics only as a secondary layer, and most arms quiet at any moment.
  Octodad is the control case for a "does this read as clumsy?" test.

### 14. Multi-leg procedural locomotion: academic + hobbyist practice
- Abdul Karim, Gaudin, Meyer, Buendia, Bouakaz, "Procedural locomotion of multilegged characters in
  dynamic environments", Comput. Animat. Virtual Worlds 24(1):3–15, 2013, DOI 10.1002/cav.1467
  (HAL: https://hal.archives-ouvertes.fr/hal-01352977). **Read:** abstract/record only.
  Four cooperating modules: character controller, **gait/tempo manager**, 3D path constructor,
  **footprint planner** → foot and pelvis trajectories; no motion data; LOD for dozens of creatures.
- Schwartz, S. A., "Creating Procedural Animation for the Terrestrial Locomotion of Tentacled Digital
  Creatures", MSc thesis, Texas A&M, 2015 — https://oaktrust.library.tamu.edu/bitstream/1969.1/155261/1/SCHWARTZ-THESIS-2015.pdf
  (403 to the fetch tool; **not read** — abstract only: real-time, physics-based, goal-directed land
  locomotion of tentacled creatures, with a production tool). **Erik: worth downloading by hand — it is
  the closest academic match to our problem.**
- Sheerin, D., "Procedural Tentacle Bundles in Edge of Tomorrow", SIGGRAPH 2014 Talks, DOI
  10.1145/2614106.2614207 (Imageworks' "Mimics": bundles that grow/retract, procedural geometry AND
  animation). **Read:** abstract only. ACM DL.
- Hobbyist consensus (PhilS94 Unity wall-walking spider, Tartu project, Godot 4.5 80.lv write-up,
  Godwalker substack — **read:** search summaries + the Godwalker post in full): a foot steps when its
  IK can no longer reach / when it drifts past a threshold from its rest target; it steps *past* the
  target (overshoot so the next step is later); neighbours never step together, legs alternate in
  groups, but a sudden lurch may break the sequence; body height/tilt follow the feet; **per-foot
  randomised step inaccuracy** to avoid machine regularity. Godwalker's lesson: the same movement code
  read "cute" or "threatening" purely from silhouette (adding spikes), and **an un-reacting locked torso
  read stiff and cheap** — body recoil on hits/shots fixed it.
- What looked bad in this genre (my synthesis of the above + common critique): metronomic stepping,
  feet that pop/slide, a body that glides at constant height while legs scurry under it (legs decorative,
  not load-bearing), and no anticipation before a direction change (Spore's authors name this as their
  biggest open problem).

### 15. The Last Guardian — Trico (genDESIGN / SIE Japan Studio, 2016); Masanobu Tanaka, "Making Trico Move", CEDEC 2017
- GameAnim write-up: https://www.gameanim.com/?p=15474 ; translated slides:
  https://docs.google.com/presentation/d/1cYDabTQGYe09w3PyUU4XLrDn2fHy9X5eJIFUjVDt89U/ ; JP slides
  https://www.slideshare.net/gendesign/ss-82621799
- **Read:** the GameAnim write-up in full (itself a rough translation of JP press + slides); slides NOT read.
- THE best-documented shipped case of a non-verbal creature that players believe has a mind.
  **Per-frame order: AI → (keyframe) animation → procedural animation → physics → fur → render.**
  Intent first, then authored motion, then procedural adjustment, then dynamics.
- **"Trico never uses fully automatic generation"** — procedural layers are "always adjusting the
  original animation". (Contrast Rain World's mostly-procedural approach; both work, but Trico's
  is the "amazed players" bar.)
- **Attention model**: pick a point of interest by per-object priority; acting on an object LOWERS its
  priority so attention moves on → visible curiosity that habituates. Action choice depends on the
  surrounding space (room → eat the barrel; cramped → rake the ground).
- **Look-at**: early version looked mechanical. Fix: **only the head locks on; the neck follows
  with a snake-like delay**; the head's rotation pivot placed **as close to the nose tip as possible**;
  head avoids walls so it doesn't clip or block the AI's line of sight. → For us: the eyes/mantle
  lock first, the arm roots follow late; a reaching arm's tip is the "nose" — pivot attention there.
- **Prediction**: for a thrown barrel Trico aims at where it *will be*. Foot landing points computed
  **at lift-off**; spine curve follows the path, slightly **exaggerated on sharp turns**; ducks under
  ceilings.
- → Directly copyable pattern: intent (target + priority decay) → base primitive → procedural
  correction (look-at with lag, contact targets fixed at lift-off) → passive dynamics.

### 16. Boucher & Dirksen (DreamWorks), "Rigging Octopuses in Penguins of Madagascar", DigiPro 2015, pp. 55–56
- PDF (moved, now 301 → dreamworks.com/research): research.dreamworks.com/wp-content/uploads/2018/07/octopus_rigging_digipro_extended_abstract_FINAL-Edited.pdf ;
  researchr: https://researchr.org/publication/BoucherD15 ; related SIGGRAPH entry "Body-Face Rig Integration on Dave the Octopus".
- **Read:** NOT read in full (PDF unreachable: redirect, Wayback blocked); content below is from
  verbatim excerpts quoted in search results only. **Erik: get this 2-page abstract (ACM DL, DigiPro '15).**
- Built on DreamWorks' existing **tail package: "layerable FK and multi-jack IK controls on a spline"**,
  extended for tentacles so animators could **slide rotational pivots up and down the tentacle**, make
  **general wave and curl shapes**, plus localized per-joint translate/rotate/scale, and "specialized
  offset controls … for grasping and localized compression … without risking changes to the tentacle as
  a whole". Face and body deformations combined specially (the mantle IS the head).
- Convergence with Pixar's Hank (independently, same year): **sliding bend sites + global wave/curl
  macros + local offsets layered over a spline.** Two studios → one pattern.

### 17. The wave principle / overlapping action as taught (animation pedagogy)
- Alex Carey, "The Wave Principle: In Theory & Practice" (Medium): https://medium.com/@alexcarey_s/the-wave-principle-in-theory-practice-1763ad5975ec ;
  Brian LeMay animation-school "Stationary Seaweed" assignment: https://brianlemay.com/Pages/animationschool/animation/1stsemesteranimation/06stationaryseaweed/06stationaryseaweed.html
- **Read:** search-result summaries only.
- A tentacle/tail/whip is "a chain of arcs, one end leading, motion moving through the object to its
  tip"; **one part arrives first and the rest catches up** (overlapping action / successive breaking
  of joints); for a self-driven appendage **the force starts at the base and energy moves to the tip**
  (dog's tail); for a passive one (seaweed in current) the drive is external. → The distinction
  matters for us: a *willed* arm motion propagates base→tip (and the biology agrees: the reach bend is
  launched proximally); a *passive* arm (drag, being hit) responds tip-first to external force. Mixing
  them up is what makes procedural tentacles look like rope.

### 18. Arrival (2016) heptapods — note only
- Sources: vfxblog Louis Morin interview https://vfxblog.com/2016/11/15/arrival-vfx-supe-louis-morin-breaks-down-the-films-key-shots ;
  editor Joe Walker, StudioDaily; Cinefex 150. **Read:** search summaries only. Vendor attribution
  conflicts (Hybride vs Framestore). Nothing on limb animation found.
- One design lesson only: the director deliberately hid the 7-limbed creatures in mist rather than
  show them fully in photoreal 3D — **concealment + slowness = gravitas**. Relevant to us only as a
  reminder that the octopus under smoke (Breach has a lit smoke layer) can be partly occluded and
  still read if silhouette and motion rhythm are strong.

### 19. Readability from a fixed high camera — Diablo III (Christian Lichtner, "The Art of Diablo III", GDC 2012) + animator practice
- Game Developer report: https://www.gamedeveloper.com/design/gdc-2012-diablo-iii-s-art-director-shows-off-design-process ;
  GDC Vault https://gdcvault.com/play/1015306/The-Art-of-Diablo ; practitioner tips (Mighty Bear Games
  "Animating for games: 4 pro tips", Medium — 403, read via search summary).
- **Read:** Game Developer report in full (fetch summary); talk not watched; Medium article via summary.
- **Legibility is the first principle** under a fixed high camera: layer background / mid-ground action
  / UI; **dim and stylise the background**; **bake high contrast lighting onto characters** — for
  giant demons, higher contrast and baked "stage lighting" **fixed depth-reading problems the camera
  caused**; strong silhouettes per archetype; larger-than-life effects; monsters given visible
  "intelligence or vanity".
- Animator practice for top-down/distant cameras: **animate and review in the in-game camera, FOV and
  distance** (not a 3/4 turntable); judge with an alpha-silhouette render from that camera; strong key
  poses beat polish; **exaggerate asymmetry** — slight changes vanish at distance; start exaggerated and
  tone down. Hades' characters were reportedly drawn a little shorter for its top-down view (mirror
  source, unverified).

### 20. Returnal (Housemarque, 2021) — GPU particle-hierarchy tentacles
- GDC 2022 VFX Summit, Risto Jankkila & Sharman Jagadeesan, "Can We Do It with Particles?: VFX
  Learnings from Returnal": https://gdcvault.com/play/1027742/ ; Housemarque blog
  https://housemarque.com/news/2021/9/15/returnal-vfx-breakdown ; 80.lv recreation (Niagara PBD/Verlet).
- **Read:** 80.lv report + 80.lv recreation article (fetch summaries); talk NOT watched.
- Enemy tentacles = **particle hierarchies** in their GPU particle system (every particle can be an
  emitter; HLSL in text files); **cylindrical meshes constructed at runtime from the particles**; same
  tech for vegetation. The recreation: parent-child distance clamp (one-sided) → two-sided PBD with
  Verlet, ends pinned, velocity re-derived from Δposition, SDF collision projected after constraints.
- → Confirms that a shipped AAA tentacle look can be "Verlet chain + runtime tube mesh" — and that
  for us the *render* tentacle (tube/skin, wiggle) and the *sim* arm can be decoupled.

### 21. Craft canon: Glen Keane, "A System for Planning and Timing Animation" (animationmeat PDF) + Lasseter 1987
- Keane PDF (read): https://eclass.aegean.gr/modules/document/file.php/511154/Theory/Glen_Keane_Animation.pdf ;
  J. Lasseter, "Principles of traditional animation applied to 3D computer animation", SIGGRAPH '87,
  pp. 35–44, DOI 10.1145/37401.37407 (not re-read; cited by Spore for its Jiggles layer); Thomas &
  Johnston, *The Illusion of Life* (1981). UW CSE457 lecture notes on principles (search summary only).
- **Read:** the Keane PDF in full (11 pages). Lasseter/Thomas-Johnston from prior knowledge.
- Keane's "Seven Essentials" relevant to us: **"Make a positive statement — do not be ambiguous"**
  (decisive poses, no mush); **"Always lead with the eyes"**; **"Animate the forces — let the momentum
  of an already animated movement suggest the next drawing; draw the leading edge of forces"**;
  **"Simplicity — what is indispensable"**. On anticipation: **"Movement — a signal of intention for the
  audience"**, **"a shape can narrow to anticipate a widening"** (an arm gathers/curls before it
  shoots out — exactly the octopus reach's pre-curl), **"anticipate with head angles and tilts"**,
  **"arcs show anticipation"**, attitudes set up by the previous attitude, timing slightly faster than
  real time ("favour the shorter length"), slow-in/slow-out vs explosive timing on ones.
- UW notes (summary): eyes lead the head, the head leads a turn; pupils start before head and body
  (≈3 frames ahead at close-up); "thinking" reads as **controlled stillness** — a longer hold on the
  decide beat; a head dip/tilt reads as judging.

### 22. Smaller notes (searched; little or nothing technical found)
- **Ursula** (The Little Mermaid, 1989; supervising animator Ruben Aquino): no interview on technique found.
  Places Erik could check: the Mouse Castle Lounge podcast (Mar 2014, two parts), the Dix Project videos,
  and the Animation Guild TAG interview. **Read:** search summaries only.
- **Clash of the Titans Kraken, the Pirates Kraken, Subnautica leviathans, HL2/Alyx barnacles, Dead Space,
  Horizon snake machines, God of War / Hades hydras, Elden Ring, Monster Hunter**: either dropped after a
  first search or not searched for lack of time. The searches I ran turned up no public technical write-up.
  Treat this as a coverage gap, not as proof that nothing exists.
- **Active ragdolls** (prior knowledge, not re-read this session): physics bodies track the animation
  through PD gains. Low gains read as drunk or comic (Gang Beasts, Human Fall Flat, the same family as
  Octodad). Shipped "serious" use (NaturalMotion Euphoria in GTA IV / RDR) keeps it to short reaction
  windows that blend back to authored motion. It is the same lesson as Spore's Jiggles: dynamics react,
  intent decides.
- **Motion matching for non-humanoids** (Ubisoft La Forge; Holden et al., "Learned Motion Matching", 2020):
  it needs a database of clips, so it does not apply to a clip-less octopus. It matters here only as the
  alternative we rule out.
- **VIPER (Angles et al. 2019), position-based elastic rods, XPBD**: left to strand (b).
- **Unreal Control Rig / Houdini KineFX / Vellum tentacle samples**: not read. They implement the same
  layering of spline IK, FK offsets and wave deformers that Pixar and DreamWorks document above.

---

# SYNTHESIS

## (1) The rig and control pattern the best work converges on

The strongest cases use the same stack in the same order: Pixar's Hank (2016), DreamWorks' Dave (2015),
ILM's Davy Jones (2006), Spore (2008), Trico (2016) and Rain World (2017). In all of them information
flows ONE way, from intent down to dynamics. Spore learned this from its "Wiggles" regression, and the
third-party rebuild of Rain World's Daddy Long Legs learned the same thing.

| Layer | What it decides | Best evidence | For our 8 × 17-bone octopus |
|---|---|---|---|
| **L0 Attention and intent** | what the animal cares about now, and how agitated it is | Trico: points of interest with a per-object priority that DROPS once acted on. Rain World: a "brain ghost", a belief about where other creatures are. Davy: one mood dial ("when Davy gets angry, the tentacles get angry") | a target plus an attention model whose priorities decay, and an arousal scalar that scales the amplitude, rate and stiffness of every lower layer |
| **L1 Action and arm allocation** | which primitive runs, which arms act, which rest | Spore "variants": game code picks the best grasper (closest, free). Hank: only 2–3 arms move. Doc Ock: one role per arm (load-bearing or delicate) | choose the acting arm(s) by geometry and role (front arms explore, rear arms anchor and walk); every other arm goes to rest or idle |
| **L2 Authored shape on a low-DOF curve** | the gesture | Hank: a few SLIDING "knots" per arm, because deformation "needs to be able to float through the geometry", drawn as a 2D curve the model snaps to. DreamWorks: sliding pivots, global wave/curl macros and local offsets on a spline. Davy: a phase-offset sine drive on the joint motors | each primitive emits a handful of parameters (a bend site s*(t) travelling outward, a curl radius, wave amplitude and phase) or a curvature profile κ(s,t), never 17 independent angles |
| **L3 Constraint and IK solve** | make the gesture fit the world | Spore's Particle IK: Jakobsen length constraints, soft (C1 response, 10 %/120 % stretch bounds), 5 outer iterations. Its **aim preconditioner** rotates and stretches the whole limb onto the goal before iterating. It blends toward the rest pose when it detects buckling, and it is **path-independent** (redundant manipulators otherwise "tie themselves in knots"). Hitman: pin a particle (invmass 0) for IK or contact. Trico: the contact target is fixed at lift-off | a particle chain per arm with planted stretches pinned; every solve starts from the primitive's shape, never from the last frame or a straight line |
| **L4 Bone reconstruction** | the 17 bones | Spore's spine splines (quintic Hermite plus transported frames); Hitman's stick figures, with twist derived for rendering; Pixar's curves → proxy meshes → final mesh | a spline through the solved knots, parallel-transport frames, 17 bones sampled from it |
| **L5 Passive secondary motion (never feeds back)** | the "alive" surface | Spore's Jiggles, passive only (the version that fed back "negatively impacted the quality of the animators' work"); Hank's linear spring on the mantle centroid; Doc Ock's cable tremor; Davy's "sticktion"; ABZÛ's vertex-shader wave masked along the body | a damped spring on the mantle and the arm roots, a small tremor, a phase-offset micro-wave per arm in the vertex shader, suckers flattening on contact, all on the render side |

Two corollaries follow:

- **At hero quality the drive can be procedural.** Simulation moved Davy's beard about 95 % of the time;
  keyframes were reserved for the tentacles that act.
- **Trico sets the bar for "amazed".** Its rule is that it "never uses fully automatic generation" but
  is "always adjusting the original animation". So the primitive (L2) has to be an authored,
  art-directed shape, and L3–L5 only correct it. Rain World shows that the cheaper opposite, mostly
  emergent motion, also works, but its quality is looser and creepier.

Cost anchors:

- Spore ran L1–L5 in **0.2 ms per 25-body creature** on a 1.7 GHz Pentium-M, 35 % of it in IK.
- Hitman used 1–10 relaxation iterations per object type. For plants, one iteration gave "exactly the
  right amount of bending".
- Davy's beard: 46 tentacles, each up to ~40 rigid segments.

How this maps to Breach (my reading, not a source): L0–L1 and the contact and anchor decisions are game
state, and L4–L5 are render-only by nature. Where L2–L3 sit is the game-state split already under
discussion.

## (2) Craft checklist: "alive and intelligent" arms (a clip passes only if every row holds)

1. **Few actors.** At most 2–3 arms do deliberate work at once; the rest rest or idle at low amplitude
   (Hank: every arm moving at once "was too hard to watch"). *Measure: per frame, count the arms whose
   joint speed is above an idle threshold.*
2. **The eyes lead.** The mantle and eyes turn to the target BEFORE any arm commits (Keane: "always lead
   with the eyes"; Trico: the head goes first, the neck lags "snake-like"). *Measure: the lead time from
   gaze onset to arm onset is > 0.*
3. **Willed motion starts at the base.** A reach bend starts at the base and travels outward, and the
   tip arrives last (Hank: lead with the "elbow", not the tip; the wave principle; strand (a)'s biology).
   *Measure: the time of peak curvature increases monotonically with arc length s.*
4. **Passive motion is the reverse.** When an arm is hit, dragged or flung, the tip lags in a whip, and
   the drive comes from outside. If willed and passive motion look the same, the arm reads as rope
   (Octodad).
5. **Anticipation.** The arm gathers before it releases: it narrows or curls before it shoots out
   (Keane: "a shape can narrow to anticipate a widening"), and the body compresses before a jet.
   Spore's authors name the lack of anticipation under sudden input as their biggest open problem, so
   plan a short wind-up even when gameplay wants an instant response.
6. **Follow-through and settle.** Every action overshoots and settles with damping; nothing stops dead.
7. **Holds and rests.** A visible "decide" beat of controlled stillness comes before the animal commits.
   Idle arms are NOT in perpetual sine motion, and there are rests between actions.
8. **De-synchrony and roles.** No two arms share phase, rate or amplitude (ABZÛ randomises phase per
   fish and rate between 50 and 150 %). Arms have roles and small offsets in temperament (Doc Ock:
   "like animating five characters").
9. **Contact is causal.** Planted segments do not slide, and speed rises with the number of gripping
   arms (Rain World). A gripping arm visibly takes load: the body sags toward it. *Measure: the drift
   of planted points is ≈ 0.*
10. **Attention habituates.** Interest in a target decays once the animal has acted on it (Trico), so it
    "looks around". Every AI state has a visible body tell, exaggerated until it reads (Rain World added
    a particle because a head turn was too subtle to see).
11. **One mood, everywhere.** Agitation changes all arms coherently (Davy's "super controller").
12. **The body answers the arms.** Arm loads move the mantle (Hank's centroid spring), and every arm
    visibly roots in the crown and web with continuous curvature (Hank's curves run on into the body).
    The Sea of Thieves kraken failed exactly here: its tentacles "connect to absolutely nothing" and
    were "mostly static".
13. **Stable over time.** The same command gives the same shape whatever the history, with no slow
    knotting or twisting over long runs (Spore's path-independence requirement).
14. **Decisive silhouettes.** Every key pose reads as a clear statement from the game camera (Keane:
    "make a positive statement"; Diablo III: legibility first).

## (3) Hank and Davy Jones: what we can copy at game cost, and what we cannot

**What we can copy:**

- Bend sites that slide along the arm instead of fixed joints (Hank's knots, DreamWorks' sliding pivots).
- Primitives defined as 2D curves (Hank's sketch-to-pose).
- Leading from the elbow, and 2–3 active arms at a time.
- Continuity from the arm into the web and mantle. In game terms: web and crown skin weights driven by
  the arm-root bones, plus a few correctives keyed on the angle between arm and body. Hank used PSD on
  exactly that angle.
- A damped spring on the mantle centroid, layered over the pose.
- Sucker stick and release as a RULE (distance to the surface plus orientation), shown by a render-side
  flatten.
- Davy's motorised chain: joints that track targets with a phase-offset sine drive, a few global dials
  and a mood scalar.
- Davy's stiction: arms cling a little to the floor and to the animal's own body as they move.

**What we cannot copy:**

- PhysBAM FEM flesh and skin simulation, and skin sliding.
- ~350 individually simulated suckers.
- Months of animator polish per shot.
- Per-shot camera cheating and staging. We have one fixed top-down camera, and an agent choosing
  actions in real time.
- PSD on dense meshes, and high-quality self-collision.

The gap has to be closed by applying the craft rules in (2) systematically, not by simulation.

## (4) Multi-limb locomotion in shipped games: what looked good, what looked bad, and why

**What looked good:**

- **Rain World.** Limbs reach ahead and stay planted until the body leaves them behind, and propulsion
  scales with the number of gripping limbs. Effort and weight emerge from that, and the body is visibly
  hauled.
- **Spore.** Each leg group has a duty factor and a phase offset, and group frequencies are harmonised
  by small rational ratios. Bodies with no feet get an inch-worm gait. Secondary jiggles are passive,
  and the whole system is cheap.
- **Trico.** A keyframed base with procedural correction on top. The contact target is chosen at
  lift-off, the path curve is exaggerated on sharp turns, and attention goes head first.
- **Davy Jones.** Sine-driven motorised chains under a few global dials.

**What looked bad:**

- **Octodad.** Passive chains led by the tip, with every limb active, read as clumsy comedy. That is by
  design, which makes it the control case.
- **Early and hobby procedural spiders.** Stepping is metronomic, the body glides at constant height
  above legs that only decorate, feet slide or pop, and nothing anticipates.
- **The Sea of Thieves kraken.** Disembodied and static.
- **Spore's own "Wiggles".** Feeding secondary motion back as IK goals degraded the authored motion.
- **Godwalker's first player rig.** A locked torso that did not react to hits read stiff and cheap.

**Why the bad cases fail**, in four words:

- **Causality**: the motion is not caused by contact.
- **Order**: no lead, no anticipation, the tip moves before the base.
- **Variety**: synchrony and regularity.
- **Embodiment**: the body does not react, and the arms have no visible root.

## (5) Readability from a straight-down camera

1. **Curvature in plan view carries the expression.** Design every primitive so its signature is
   visible in XY: curl direction and radius, S-bends, spread, the travelling bend. A motion whose only
   content is vertical (an arm raised to strike) needs an XY tell, such as a wind-up curl or a sweep.
2. **Height is invisible without cues.** Use the projected shadow (Breach's #114 unit-shadow path),
   with its offset growing with height. Add contact darkening or AO where an arm touches, sucker
   flattening on planted stretches, and a little extra specular or rim light on raised arms.
3. **Contact and free must read at a glance.** It is the main locomotion signal (Rain World, Hank's
   suckers, Davy's stiction).
4. **Review in the game camera at game zoom**, with an alpha silhouette render from that camera. Strong
   key poses beat polish. Exaggerate amplitudes and asymmetry beyond the biology, because slight changes
   vanish at distance; start exaggerated and tone down (practitioner advice; Diablo III).
5. **Contrast.** Light the creature with high contrast against a background that does not compete.
   Diablo III's baked "stage lighting" fixed depth reading under a fixed high camera.
6. **The eyes sit on top of an octopus, so use them.** Make the gaze direction legible from above (the
   eye bulge turning, the orientation of the pupil bar) and exaggerate it, because gaze is how the
   player reads intent (Trico, Keane, Rain World's cue particle).
7. **One hero arm at a time** gives the eye a target. With 8 arms, focus is everything (Hank).

## (6) Sources Erik should get (I could not read them)

**Papers and abstracts** (ACM DL unless noted):

- Schwartz, S. A., "Creating Procedural Animation for the Terrestrial Locomotion of Tentacled Digital
  Creatures", MSc thesis, Texas A&M, 2015. Open access at oaktrust.library.tamu.edu, but my fetch tool
  got a 403. **Highest priority**: it is the closest academic match to our problem.
- Boucher & Dirksen, "Rigging Octopuses in Penguins of Madagascar", DigiPro 2015, pp. 55–56 (the
  DreamWorks PDF now redirects), plus "Body-Face Rig Integration on Dave the Octopus" by the same authors.
- Criswell, Derlich, Hatch, "Davy Jones' beard: rigid tentacle simulation", SIGGRAPH 2006 Sketches,
  DOI 10.1145/1179849.1179995.
- Sheerin, "Procedural Tentacle Bundles in Edge of Tomorrow", SIGGRAPH 2014 Talks, DOI 10.1145/2614106.2614207.
- Abdul Karim et al., "Procedural locomotion of multilegged characters in dynamic environments", CAVW
  24(1):3–15, 2013, DOI 10.1002/cav.1467 (HAL hal-01352977 may have a free copy).
- Coleman & Kautzman, "Stable, art-directable skin and flesh using biphasic materials", SIGGRAPH 2012
  Talks (background to Hank's sim; low priority).
- Lasseter, "Principles of traditional animation applied to 3D computer animation", SIGGRAPH '87,
  DOI 10.1145/37401.37407.

**Talks** (by title; free ones marked):

- Jakobsson & Therrien, GDC 2016 Animation Bootcamp talk on Rain World's procedural animation. **Free on
  YouTube.** Worth watching, because my tool could not get the captions.
- Rosen, "Animation Bootcamp: An Indie Approach to Procedural Animation", GDC 2014. **Free.**
- Tanaka, "Making Trico Move", CEDEC 2017. **Free slides**: Japanese on SlideShare, plus an English
  Google Slides translation.
- Nava, "Creating the Art of ABZÛ", GDC 2017 (Vault).
- Jankkila & Jagadeesan, "Can We Do It with Particles?: VFX Learnings from Returnal", GDC 2022 VFX
  Summit (Vault).
- Lichtner, "The Art of Diablo III", GDC 2012 (Vault).
- Murphy & Tibitoski, "There and Dad Again", GDC Next 2014. Octodad; only if we want the negative case
  documented.

**Books and print:**

- Jones, *Digital Creature Rigging: Wings, Tails & Tentacles for Animation & VFX* (Routledge, ISBN
  9781138560697). The long form of the rigging pattern.
- The Cinefex issues on *Dead Man's Chest* (2006) and *Arrival* (Cinefex 150), and *The Art of Finding
  Dory* (2016). Not verified whether any of them covers the rigs.

## Honesty ledger

**Read in full:**

- Talbot et al. 2016 and Kautzman et al. 2016.
- Hecker et al. 2008 (all 11 pages).
- Jakobsen 2001.
- Glen Keane's PDF.
- Both CGW Davy Jones articles and the CGW Doc Ock article (via fetch summaries).
- The GameAnim write-up of the Trico talk.
- The Godot doc on the ABZÛ technique.
- The Rain World devlog: about 60 % of its ~786k characters, via fetch summaries.
- The short press pieces on Hank.

**Secondary or abstract only:** DreamWorks DigiPro, the Davy SIGGRAPH sketch, Edge of Tomorrow,
Schwartz, Karim et al., Octodad, Sea of Thieves, Carrion, Returnal, Diablo III, Arrival, Rosen's talk,
and the Rain World GDC talk itself.

Items marked "prior knowledge" or "my analysis" are not sourced this session.
