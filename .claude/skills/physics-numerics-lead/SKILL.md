---
name: physics-numerics-lead
description: Claude's standing role as lead of Breach's physics numerics -- how the gas, energy, water and blast solvers are computed (time stepping, splitting, stability, units, limiters, the cost per tick), as distinct from what the physics models, which is Erik's. Holds the mandate, the principles learned from incidents (each beside its incident), the diagnostic toolkit, and where the open numerics work lives. Load at the start of ANY session that touches a solver's stepping, substeps, limiters, caps, drag, dials in physical units, a numerical instability, unrealistic winds or temperatures, or arc #217, before reading the code.
---

# Physics numerics lead -- continuity skill

**Status: PROVISIONAL** (created 2026-11-21, during #217's blast-stability
fix). Every session that uses it amends it before merging: what moved, what
was decided, what was learned. A step is a MUST only with its incident beside
it, and is demoted the same way.

## 1. The mandate (Erik, 2026-11-21: "i'll leave it to you")

The master CLAUDE.md splits the calls this way:
- **Claude decides** the computational method: time stepping, operator
  splitting and its order, substep and CFL logic, limiters, solver schedules,
  fixed-point arithmetic, CPU/CUDA structure, cost.
- **Erik decides, always**:
  - the physical model: what the air, water and fire DO;
  - every dial judged by feel (`k_drag2`, blast strength, vent rates);
  - calibration targets;
  - whether a change in behaviour is acceptable: a ceiling that moves knockdown,
    energy returned as heat instead of destroyed;
  - budget.
- **Default**: a numerical fix that changes a calibrated behaviour is a
  numerics call, but the behaviour change goes to Erik as a ruling, with its
  default stated.

Erik's goal: smoke and air that move believably for atmosphere, without solver
noise. Training speed matters to him as much as looks.

## 2. Principles, each with its incident

1. **Every quantity in physical units: m, s, kg.** Never tiles or ticks.
   Check a law at two tile sizes (1.0 m and 0.333 m). A law whose result
   changes with tile size has a hidden `dx`.
   *Incident: #217 patch U, 2026-11-20. Transport moved u·dt TILES while the
   rest of the EOS used metres, so at 0.333 m air moved at 1/3 of the wind.
   Erik tuned `k_drag2` at 1.0 m and then retuned at 0.333 m, unknowingly
   compensating.*
2. **A stability condition holds for the WHOLE step, not just the part that
   is substepped.** Kwatra 2009 and every ICE-type scheme are stable at any
   acoustic CFL only if the whole step meets the advective CFL (air moves at
   most about one tile per pressure update).
   *Incident: #217 §8. From the first solver (eos-p3, 2026-07-10) the 8
   substeps wrapped transport only, and the pressure was solved once per tick.
   Once patch U made transport honest, air crossed 2-36 tiles per solve, cells
   emptied and piled to 113× ambient, and empty cells pinned the solved P.*
3. **Test with each limiter and drag OFF.** A damper that holds a scheme up
   hides an instability, and every later retune fights it.
   *Incident: #217. With `k_drag2 = 0` the quiet arena ran to 1000 m/s in
   4 s. Erik had felt the drag tuning was "close to flipping out".*
4. **No clip between a solve and the energy books, even a counted one.** A
   cap that exports energy the step itself put in drains real heat while the
   books "close".
   *Incident: #217 §8. The kick debited KE at the uncapped speed and the cap
   exported it, so blast rooms cooled by about 56 K and the mass-weighted
   T went below ambient.*
5. **A calibration made under a bug is suspect once the bug is fixed.** List
   every dial tuned in the bug's regime and hand the list to Erik.
   *Incident: #217. The `[blast] expansion_m3_per_strength`, the frag row
   (40 m³ / 10 MJ against about 2.6 m³ / 0.9 MJ physical), the studio charges
   (156 MJ) and `k_drag2` were all set while transport ran at 1/3.*
6. **Cost is measured at the training batch size, never at N = 1.** The
   batched engine runs together, so a per-env need becomes a batch-max cost
   unless the design masks it per env.
   *Incident: #217 lens 2. Batch-max k measured +55-88 % per tick at N = 64,
   against +14-43 % with per-env k.*
7. **Prove a cause on the live path. An inference stays labelled until
   measured.**
   *Incident: #217. The patch-U session inferred "donor-cell transport above
   Courant 1"; it was REFUTED (N_SUB_MAX 48 kept Courant ≤ 0.25 and the
   second charge still went wild).*

## 3. Diagnostic toolkit

- **Is it numerical or physical?** Raise the tick rate: 4-8× more pressure
  updates on the same scene. If the effect vanishes, the scheme made it.
  #217: the quiet arena went from 35-54 m/s at 24 tps to 0.4-0.6 m/s at
  96 tps.
- **The quiet scene is the clean test.** A sealed room with no source should
  hold near-zero wind. Use `levels/test_arena`, `tests/_quiet_books_bench.py`
  and `tests/_sealedbox_bisect_bench.py`.
- **The instruments** (extend one, never write a parallel one):
  - `tests/_eos_checkerboard_bench.py`, including `--stability`, `--steps`,
    `--table` and `--solve-bias` (on branch `217-eos-checkerboard` until it
    merges);
  - `tools/wind_drift_probe.py` (plume drift / wind);
  - `tools/e2e_drive.py`;
  - `tools/bench_conductor.py`, staggered across envs, for training cost.
- **Kinetic-energy budget per tick**: pressure work in, drag out, cap out. It
  shows which term feeds a runaway.
- **Two independent agents work well:** one measures on the live path, one
  reads the code only as a numerics reviewer, and neither gets the other's
  hypothesis. Then the measurer confirms or refutes each review finding. Run
  #217 2026-11-20: the review found the cap heat drain and the empty-cell
  sink, and the measurer refuted the old inference.
- **Source realism table:** compare every wind source with a back-of-envelope
  figure: vent grille 1-3 m/s, room fire plume ≲ 1-5 m/s, a frag about
  0.9 MJ.

## 4. Where things live

- Solver: `cpp/src/eos_solver.{h,cpp}` with its CUDA twins
  (`cuda_eos_step.*`, `cuda_eos_resident.cu`, `cuda_kick_compression.cu`,
  `cuda_resident_tick.*`), transport `cpp/src/bulk_transport.cpp`, and
  orchestration in PhysicsEngine and `src/simulation/physics_runner.py`.
- Rules: `.claude/rules/energy-books.md`, `fields-gas.md`,
  `device-residency.md`, `determinism-gates.md`.
- The arc: #217, doc `docs/eos_checkerboard_design_2026-11-20.md` on branch
  `217-eos-checkerboard`:
  - §3: the checkerboard's cause;
  - §7: patch U;
  - §8: blast stability, with the numerics review in §8.3 and the critiques
    in §8.15.

## 5. Open numerics work (2026-11-21)

1. **The stack on `217-eos-checkerboard`, in order:**
   - U (built);
   - §8 whole-step sub-ticks with per-env k, a ceiling of K_MAX·dx/dt and
     cap energy returned as heat. Erik said yes to all five rulings on
     2026-11-21; it is being built, and U + §8 merge together;
   - (e) the multigrid rounding bias;
   - (d)+(a): finer P resolution plus Rhie-Chow, the checkerboard cure;
   - gas diffusion in m²/s.
2. **The drag session with Erik after §8 lands.** The §8 build audits the
   drag law's units and measures what drag still does once the ceiling is in.
3. **A blast and vent source session with Erik**, using the source table in
   §8.4. Vents also drain their rooms (a constant mass flow in tile volumes).
4. Not cured by §8: F5 (the kick floor and the unweighted face mean let
   near-empty cells drive their neighbours' mass) and F7 (momentum is not
   conserved).

## Run log

- 2026-11-21: created from #217's §8 phase (diagnosis, independent numerics
  review, two critics, v2, implementation brief).
- 2026-11-21: #217 §8 built (whole-step sub-ticks, per-env k, the budget
  ceiling, the cap heat; design §8.16). Learned: (a) the A/B baseline must be
  captured on the base build BEFORE the first code commit, and every switch
  of a new mechanism needs its "legacy == HEAD" guard committed -- the
  three-switch T6 found nothing, which is the point; (b) a full suite on
  this PC is ~1 h 40 min serial and shares the card with other sessions:
  run targeted tests while building and ONE full suite at the end (Erik,
  2026-11-21), shard by file if forced to run one; (c) "test with each
  limiter off" paid again: `k_drag2 = 0` stays stable under the ceiling, so
  drag is no longer the hidden limiter (design §8.16, "Drag after §8");
  (d) a rule from the entry state lags an in-tick kick by one tick -- a
  1 / 2 limit cycle in k after a blast, and the legacy studio rows (heat with
  no divergence source) get k = 1 for two ticks. Open: the ring-adjacent
  arena winds (outside the k-set by design), the breach's k tail on
  residual pockets.
