# Octopus motion prototypes (P1–P5 of the animation study, #105)

Pure numpy, no game, no Blender, no raylib: the map-editor pattern (pure core,
thin shell). What survives is ported to C++ Q16 in P6; nothing here is engine
code and nothing in `src/` or `cpp/` imports it. Study brief:
`docs/octopus_animation_study_2026-10-29.md` §5.

Run with this machine's breach Python (`docs/dev_setup.md`):

    <breach-py> prototypes/octopus_motion/reach_gates.py     # P1, ~2 min, writes out/

## P1 — one arm, the make-or-break (2026-10-29): PASSED, 7/7 gates

`arm_chain.py` — an XPBD chain of 18 nodes (one per game bone + tip),
tapered mass, 17 distance + 16 angle constraints, actuated by Yekutieli's
stiffening wave (a sigmoid activation front travelling base → tip at
`L / t_act`; activation × a tapered bend stiffness), anisotropic quadratic
drag (`c_per 1.013 / c_tan 0.0256`, measured on a real arm), 16 substeps × 2
Gauss-Seidel passes per 24 Hz tick.

`reach_gates.py` — the experiments and gates; `out/report.md` the numbers,
`out/fig*.png` + `out/reach.gif` the pictures.

What it showed (Yekutieli's 240 mm arm, a_ca 0.6, t_act 0.9 s):

- **The reach is reproduced.** One bend travels base → tip at the wave's own
  speed (peak / wave 1.03–1.18 across the natural t_act range), with the
  three-phase (rise, plateau, decrease) bell velocity; the arm behind the
  bend is straight (≤ 2.5° mean local angle); the tip ends straight within
  1–2 mm; the bend dissolves at the tip.
- **Kinematics are independent of force** above a minimum-force curve
  (normalised profiles within RMS 0.07 for a_ca 0.2–1.0; peak speeds within
  7 %), and the map in (t_act, a_ca) has Yekutieli's shape: a knee at low
  force and a wall at t_act ≈ 0.25 s where no force propels the bend and the
  whole arm opens at once (the startle look). Too little force = a lazy
  curved reach. Both failure looks are authored by the same two dials.
- **Drag is load-bearing.** The same command without drag overshoots the
  tip by 300° of distal turn (Yekutieli's finding); with it, 0.5°.
- **A dead arm** (no activation, whipped about the base) decelerates
  monotonically and stays curved — the passive signature for #118.
- **Solver resolution, not force, sets straightness.** 8 substeps × 1 pass
  leaves the arm behind the bend 6–7° curved however stiff the muscle
  (k_bend × 4 changed nothing); 16 × 2 or 32 × 1 brings it to 2°. Passes
  and substeps are not interchangeable here: 32 × 1 whips the distal arm
  (−48° mid-reach S-flex), 16 × 2 does not (−12°).
- **Cost (gate d), the open risk for P6:** ~2.5 k mul/add + 184
  reciprocals + 34 sqrt + 32 atan2 per arm-substep; × 16 substeps × 8 arms
  ≈ 0.6 M weighted ops per octopus-tick, i.e. ~0.6 ms at 1 ns/op against the
  0.2 ms ceiling. P6 must measure; the levers are a direct (tridiagonal)
  bend solve per arm, a sin-based bend constraint without atan2, fewer
  substeps once the game-scale arm (8× longer, same durations) is tuned, and
  SIMD over the eight arms.

Not in P1 (by design): the base rotating during the reach (why real paths
curve), elongation (`stretch`), 3D, contact, suckers. Those are P2–P4.

## P2 — the crawl (2026-10-29): 5/6 gates, the sixth a narrow miss

`arm_bank.py` — the P1 chain batched over A arms (bit-identical for one arm,
gate 0), plus sucker anchors with a grip limit, floor friction, rest-length
scaling (elongation), the linear-drag control and a rigid `Body2D` the arm
roots attach to (PBD rigid attachment: the body gets ONLY the arms'
reactions and its own drag). `crawl_sim.py` — eight arms on a deck seen from
above, Levy 2015's PUSH_STEP (mid-arm sucker patch anchors, the proximal
segments elongate to 1.6× over 0.5–1 s, release on saturation) and the hub's
seeded lottery (no clock; alignment × hind-arm bias × neighbour bonus; ≤ 3
in stance; steering = aiming the lottery at wanted + (wanted − recent
travel)). `crawl_gates.py`, `out/report_p2.md`, `fig6`, `fig7`, `crawl.gif`.

- Travels within 2–3° of any wanted direction, 0, 70, 135, −90° off the
  heading, at 0.05 m/s mean / 0.1 m/s peaks (Levy: ~0.1 m/s peaks), 1–3 arms
  pushing (mean ~1), never more.
- **No rhythm, and the test can fail:** the whitened speed spectrum's
  highest line is 6× its neighbours (chance level for 76 bins); the same
  animal on a clock (no jitter, recruit the instant an arm is ready) shows a
  41× line at 1.45 Hz.
- 7 arms: 72 % of the 8-arm distance, 2° off; 6 arms: 90 %, 8° off. No
  special case for a dead arm: it is simply never READY.
- Heading wanders ±25° over 25 s and never turns to face the travel
  direction (the real animal: ~25° SD per 2.8 s).
- **The miss:** Levy's per-push statistic, circular r between the pushing
  arms' summed direction and the body's displacement during that push, is
  0.88–0.94 against the brief's 0.9 (per-push error 15–19°). The body
  coasts between pushes (momentum from the last push colours the next) and
  the steering alternates sides; without steering the per-tick cosine was
  0.93–0.95 but the 7-arm animal veered 24°. Left as a FAIL in the report;
  the eye (P8) and the port (P6) decide whether 0.88 is a defect.

## P3 — swimming (2026-10-29): 3/3 gates

`swim_sim.py` — side view, two bank arms (the up / down groups of four,
mass and drag ×4, the web as a ×3 broadside drag factor), the umbrella
(root angle commanded on the body, 20–80°, open 1.0 s / close 0.4 s, stiff
paddle) and the jet (a body impulse of 0.8 m/s over 0.3 s, arms bundled).
`swim_gates.py`, `out/report_p3.md`, `fig8_swim.png`, `swim.gif`.

- **Travel from drag alone, and only from its nonlinearity:** 0.036 m/s
  steady (51 mm per stroke) with the real quadratic anisotropic drag; the
  control with linear isotropic drag on arms and body nets 5 % of that (it
  oscillates in place).
- A jet carries 5.8 mantle lengths and decays to 10 % in 2.9 s.
- A jet train (one per second, arms bundled) does 0.28 m/s = 5.6× the
  crawl; the umbrella alone is 0.7× (slow in the animal too: the cruise);
  umbrella + half-jets 2.4× because the open umbrella brakes. Gameplay
  speeds remain Erik's dial.

## P4 — reach, grab, hold, carry a prop (2026-10-29): 3/3 gates

`grab_sim.py` on the crawl animal: the prop is a 0.1 kg disc (a marine is
~0.3 of the octopus) placed out of reach at 50°; REACH = aim the root in the
body frame + the P1 stiffening wave + stretch to 1.4×, stopping on contact;
GRAB = rigid attachments (the bank's `grabs`, the same PBD attachment the
roots use) between the touching mid-arm nodes and the prop; the WRAP sets
curvature targets distal of the first seal, toward the prop, sealing what
touches; HOLD pulls the proximal segments in to 0.7× at full stiffness;
CARRY = the P2 crawl with the holding arm in a HOLDING state the lottery
never sees, then three jets in the plane. `grab_gates.py`, `out/report_p4.md`,
`fig9_grab.png`, `grab.gif`.

- Contact 0.29 s after the reach begins, at s = 0.47–0.53 L; after the wrap
  5 nodes sealed at 0.41–0.65 L. Mid-arm, never the tip.
- Attached throughout: 5 sealed, 0 pops, through 15 s of loaded crawl and
  three 0.5 m/s jets. The hold is LOOSE — the prop swings −7..+47 % of its
  87 mm distance during the crawl and −16..+49 % through the jets (reported,
  not gated: tuning for P6/P7, not a drop).
- The others re-plan: L1 never pushes after the grab; the loaded crawl runs
  at 110 % of the unloaded speed (seed-to-seed noise), 1° off wanted.
- "Lift" is out of this plane; here the pull-in stands for it. The z of the
  chain is the brief's open question 2.
- Side effect on P2: `_body_predict` now applies the deck friction to the
  body too (needed for the prop); the P2 numbers in `report_p2.md` predate
  it (in water the body's friction is ~2 % of its weight; negligible).

## P5 — idle (2026-10-29): 5/5 measurable checklist rows

`idle_sim.py`: the seated animal (every still arm's mid-arm patch sealed to
the deck); with no travel goal the hub lets at most two arms EXPLORE at a
time: a decide hold (0.4–1.6 s of stillness), a reach (the P1 wave, aim
±45° in the body frame, stretch 1.15–1.35), a touch (0.3–1.1 s), then LIMP
(activation off, the arm relaxes under drag) and it sticks down again where
it lies. Seeded, no clock, front pairs favoured. `idle_gates.py`,
`out/report_p5.md`, `fig10_idle.png`, `idle.gif`. Breathing and tip curls
are render controls (CONTROLS.md), not scored here.

- Row 1 few actors: at most 2 arms' distal halves move at once (the
  proximal segments of seated arms jiggle at ~3 cm/s when a reach turns the
  body: secondary motion, counted out on purpose).
- Row 3 willed motion from the base: in 95 % of gestures the time of peak
  speed increases monotonically with node index (Spearman median 1.00).
- Row 7 holds: still 38 % of the time, longest hold 2.7 s; onset gaps
  2.5 ± 1.7 s (CV 0.68). The P2 spectral-line test is reported but does not
  bind here: 24 events and gestures ~4 s long give a broad bump at 0.2 Hz
  that the whitening window cannot flatten.
- Row 8 de-synchrony and roles: 2 of 24 gestures start within 0.3 s of
  another arm's; L1 7 gestures, the back pair 1 each; every arm moved.
- Row 13 stable: still arms' segments at rest length to 0.0 %, body drift
  0 mm in 60 s (the anchors hold it; before they were added a reach's
  reaction moved the body 36 mm and every arm with it).

The rows judged by eye (2 eyes lead, 4–6 passive/anticipation/settle, 9–12,
14) wait for P7's render and P8's clips.

## P5z — the chain at GAME scale, in 3-D (2026-10-30): 33/35 gates

The float step the P6 design (section 6.8) puts before the integer reference:
the P1-P5 gates re-run on the MID octopus in 3-D, with z, a floor and a
Bishop frame, and the row values by a stated law. Nothing here is engine code.

    <breach-py> prototypes/octopus_motion/gates3d.py     # ~4.5 min on 16 cores (25 scenarios in parallel), writes out/

Files: `arm_chain3d.py` (the 3-D chain: drag, g_eff, distance, Bishop frame,
two signed bends per node, floor, the tabulated activation), `arm_bank3d.py`
(batched bank + bodies with z + anchors + grabs; `BankView` / `BodyView` give
the 2-D hubs their interface, `World` steps several animals at once),
`crawl_sim3d.py` / `swim_sim3d.py` / `grab_sim3d.py` / `idle_sim3d.py` (the
2-D hubs INHERITED, not copied), `game_scale.py` (`mid_row()` writes
`out/row_mid.toml`, the hand-off to P6a), `scenarios3d.py` (every run as a
parallel job, with replay traces), `gates3d.py`. Out: `report_p5z.md`,
`row_mid.toml`, `fig11_*.png`, `{reach,crawl,swim,grab,idle}3d.gif`,
`traces_p5z.npz` (10 MB: reach, dead arm, free animal, crawl, jet, grab, idle
-- node positions per tick + the solver's per-tick inputs; layout in its
`README` key). Two 2-D files got behaviour-neutral seams:
`CrawlParams.steer_min_dist` (was a 0.02 literal) and `Swimmer.tick` split
into `_command` / `_log`.

**The law** (same-duration similarity, S = 2.52 / 0.24 = 10.5): lengths S,
masses S^3, k_dist S^3, k_bend S^5, grip limits S^4, torque damping S^5,
c_per / c_tan unchanged (the areas are recomputed from the scaled radii),
durations 1, speeds S (jet_dv 0.8 -> 8.4 m/s, the carry jets 0.5 -> 5.25 m/s),
g_eff NOT scaled (0.188 m/s^2). **No dial was tuned**; the solver stays
16 x 2 at 24 Hz (no gate forced a change).

**Where the row leaves the law** (each flagged in the TOML): body mass 510 kg
from the SIZES mantle (a 0.93 x 0.93 x 1.08 m ellipsoid at 1042 kg/m^3; the
law: 347 kg); crown radius 0.465 m = half the mantle width (law 0.525 m); the
body's drag areas from that radius. The arm RADII follow the law (base
0.079 m) although SIZES draws the mid arm 0.30 m thick at the root (radius
0.15 m: ~1.9x thicker, ~3.6x the arm mass) -- a ruling for Erik / P6a; the
prototype keeps the law so the scale test stays clean. The prop is 116 kg by
the law (the game's marine is 90 kg). g_eff unscaled means gravity AND the
deck friction (mu g) are 10.5x weaker relative to the animal's own
accelerations than at 0.24 m. New dials (not in 2-D): `lift_pitch_deg` 30,
`lift_s` 1.0.

What it showed:

- **The law holds exactly where gravity does not enter.** The 3-D chain equals
  the 2-D P1 chain in a plane to 1.8e-14 m (gate 0); the same reach in a tilted
  plane, rotated back, to 8e-14 m (gate 0b: the Bishop frame is right). P1's
  gates at mid scale reproduce the 2-D numbers digit for digit (peak/wave
  1.03-1.28, gate-b RMS 0.098-0.041, the drag-free control overshoots by
  257 deg): the dimensionless system IS the same. With g_eff on and the base
  1.5 m up, the arm sags 1 cm during a reach.
- **The tabulated activation** (1025 samples of S(x) = (1 + tanh x)/2 on
  [-8, 8], step 1/64, linear interpolation, max error 1.2e-5) gives the same
  P1 gates to the printed digit as tanh.
- **The crawl at 0.6 m/s** (2-D x S = 0.53), within 1-3 deg of every wanted
  direction, 1-3 arms pushing, no rhythm (lottery 8.6x vs clocked 77.5x),
  98 % / 100 % of the distance with 7 / 6 arms, never off the deck. FAIL 1 is
  the known 2-D near miss (Levy's per-push r 0.87-0.94; not bent). **FAIL 5 is
  new**: heading drift over 25 s +9 / -1 / +25 / +59 deg (limit 30). A
  sensitivity sweep showed seed-dominated wander (dir -90, seeds 1/2/3:
  +59 / -1 / -20 deg; dir 135: +25 / -46 / +34) that none of three candidate
  dials removes: mu x S (the 2-D's relative friction back) +47; the law's body
  (347 kg, 0.525 m) +47; yaw damping x10 +23 / -2 / -12 at -90 but +34 at 135.
  Nothing adopted. The body wanders more at game scale (weaker relative
  friction, heavier body) but never turns to FACE the travel direction (+59
  while travelling -90 is away from it). Whether 30 deg per 25 s is the right
  bound at game scale (the real animal: ~25 deg SD per 2.8 s) is P8's eye.
  Side observation: travel 0 and travel -90 on the same seed are exact
  rotated copies of each other (the hub is symmetric under that rotation).
- **Swimming in the real 3-D chain** (eight arms in a cone, the umbrella a
  root angle about the swim axis): umbrella 0.41 m/s from drag nonlinearity
  alone (the linear isotropic control 7 %), a jet carries 7.8 mantle lengths
  and decays in 3.8 s, the jet train 3.95 m/s = 6.5x the crawl. The body sinks
  ~0.07 m/s under g_eff while it swims (reported; holding depth is a future dial).
- **Grab + LIFT**: contact mid-arm (0.53 L) 0.29 s into the reach, 5 sealed
  at 0.41-0.65 L, 0 pops through 15 s of loaded crawl and three 5.25 m/s jets;
  the holding arm's root pitched up 30 deg over 1 s lifts the 116 kg prop to
  0.20-0.35 m and it never touches the deck again. The hold is loose
  (-51..+64 % swing, reported as in P4).
- **Idle**: rows 1/3/7/8/13 pass with the 2-D's numbers (body drift 0.6 mm in 60 s).
- **N1**: a dead arm released 0.2 m above the deck touches it at 1.7 s, lies
  flat by 2.4 s and is motionless over the last 5 s of 20 s. Reported beside
  it: the curled dead arm UNFURLS on its passive stiffness and its centroid
  travels 2.5 m over the deck before friction stops it -- passive
  straightening (1 % of k_bend toward a straight target) is strong against
  mu g at game scale. A dead arm probably wants its own pose as the passive
  target (#118's question).
- **N2**: the free animal (no anchors, mu 0, idle waves on, the "symmetric"
  uniform-rate drag v <- v/(1 + dt/tau) on every node and the body, under which
  the centre of mass cannot move in exact arithmetic): COM 2.5e-13 m in 60 s
  while the body swings 4 cm. That is the float baseline the integer port's
  rounding must match. With the REAL quadratic drag the same idle animal's COM
  travels 3.8 m in 60 s -- drag-nonlinearity thrust, P3's physics, not slide:
  the integer no-slide gate must use the symmetric drag or it measures swimming.

Lessons for the integer port (P6b):

- The frame transport needs no trig: nA' = nA - ((nA . t') / (1 + t . t')) (t + t'),
  one division per segment (shared by both normals), then Gram-Schmidt against
  t' and a normalise (one sqrt); nB = nA x t. Straight is the EASY case (the
  correction is exactly 0); the only singular case is a 180-degree fold
  between two segments (1 + c = 0), floored at 1e-9 here -- the port should
  floor 1 + c at a few counts. Frames are rebuilt from the root every pass and
  never carried across ticks, so rounding cannot accumulate along time.
- Magnitudes at mid scale (dt = 1/384 s): node mass 1.42 kg at the base,
  13 g at the tip (18.3 kg per arm); at the tip the passive bend has
  alpha = 1/(k dt^2) ~ 6.4e5 against W ~ 7.8e3, so its dlam is ~1 % of C:
  the stiffness-form numerator k C must be pre-shifted enough that the
  passive tip does not round to zero; at the base the active bend has
  alpha ~ 23 against W ~ 1.2e2 (W dominates).
- Plane A of a deck-lying arm is an exact no-op in float (the out-of-plane
  coordinates are exactly 0); in integers they will be +-1 count, so plane A
  sees ~1e-7 rad of noise per node -- harmless if rounded symmetrically, and
  the no-slide gate is what will notice a bias.
- The free-animal COM is 1e-13 m in float: integer drift above a few counts
  per substep is rounding bias (design 6.3), never physics, provided the
  symmetric drag is used.
- The activation table: index = (x + 8) * 64 is a shift of a Q16 argument;
  clamp before indexing; its 1.2e-5 error is below one Q16 count (1.5e-5).
- Solver resolution: 16 x 2 is still right at mid scale (exact similarity:
  every dimensionless solver number -- k dt^2 / m, drag dt K v / m -- is the
  0.24 m animal's), so P1's 8 x 1 / 32 x 1 findings carry over unchanged.
