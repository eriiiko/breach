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
