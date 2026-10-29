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
