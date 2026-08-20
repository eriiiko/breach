# HUMAN-TEST 2026-08-20 — P-V3 velocity clamp: **PASS**

Erik played `playground` on the P-V1 build (branch `velocity-clamp`,
commits `aabb9f4..a9c08ac`, CUDA, Lenovo), pressure visualization on —
the same scenario shape that seeded the arc.

**Erik's verdict:** *"feel test is perfect, we can push and merge."*

Expectation had been set from P-V2's measurement (spikes attenuated
~30–40%, not necessarily gone; the N_SUB_MAX=8 substep rail is the
remaining owner of residual flashing). The feel cleared the bar outright.

Gate summary behind the verdict (full numbers:
`docs/velocity_clamp_pv2_measurement_2026-08-19.md`):

- own-cell cap violations: 52,923 cell-snapshots (Mach 2.47) → **0**
- P_min −1.324 → −0.310 atm; worst cell 433× → 299× ambient;
  peak single-tick pile-up 328 → 197 cell-eq
- CPU↔GPU lockstep tol 0; `u_max_hits` structurally 0

**Consequence:** merge to main approved; the one-time golden re-baseline
(six standing digest reds + the 11 GOLDEN_AGGREGATE flips) executes at
this close per Erik's standing ruling. The N_SUB_MAX question opens as
its own item with P-V2's numbers.
