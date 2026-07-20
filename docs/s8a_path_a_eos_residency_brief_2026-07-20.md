# S8a Path A — EOS device residency (Fable brief, 2026-07-20)

**For:** Fable (you wrote the S8a residency plan). **From:** Opus, mid-build.
**Status:** DESIGN BRIEF — start with a design pass + adversarial critique, THEN build.
This is the determinism-critical half of S8a that the residency spec deliberately deferred
(`cuda_mg_solve.h` §2.7: on-device MG-build is *"the S8 endpoint … not P6"*).

## Where you're dropping in

Opus built **Path B (Rung 1)** first: the residency *framework* — persistent CuPy-owned device
fields, `step_resident(...)` orchestrator, per-solver `*_launch_resident` launch cores for the
**leaf** solvers (water/smoke/fire/temperature/combustion), the GameMap CuPy residency mode
(`device_ptrs()` / batched `to_host()`/`from_host()` / `__setattr__` guard), and the `--resident`
flag. See `docs/cuda_s8a_residency_spec_2026-07-19.md` (the ★ BUILD FINDING block) + the merged
Path-B commits on `cuda-s8a-residency`/main.

Path B keeps the EOS stage on its current host-island path and **brackets it with one batched D2H
(before EOS) + one batched H2D (after EOS)**. **Your job: eliminate that bracket** by making EOS
fully device-resident, so the fields never leave the GPU across the whole tick — the real §3.3
"zero mid-tick transfers."

## The EOS host island you must port (read `cpp/src/cuda_eos_step.cu`)

Today, inside `eos_step_cuda`, after the device SL-advection + bulk-flux substep loop:
1. **Mandatory D2H** at the substep/solve boundary (`cuda_eos_step.cu:~360`) — pulls wind_x/wind_y/
   temperature/gas planes back to host for the host digests + the solve inputs `mg_build_levels`
   consumes.
2. **Host reductions** (verbatim from `eos_solver.cpp::step`, per the P6 review §1.6): `div_u`,
   the Dalton/`pstar` pressure assembly, `c_local` + the substep-count reduction. All the `_host`
   helpers near the top of the `.cu` (`mul128_shr_host`, `mirror_idx_host`, etc.).
3. **`mg_build_levels` on the host** (`eos_solver.cpp`): builds the whole multigrid hierarchy —
   level-0 operator, **Galerkin coarse operators** (R·A·P triple products per level), **Q.32
   diagonal reciprocals**, and the **P_prev warm start**. This is the substantial gather-heavy port.
4. **`eos_mg_vcycle`** + **`eos_kick_compression`** (`cuda_mg_solve.cu` / `cuda_kick_compression.cu`)
   — already kernels, but they take **host** pointers and do their own internal H2D/D2H (their
   headers say "PERF NOTE (residency is S8)"). Rework to take device pointers from the resident set.
5. **Host materialization** of `atmosphere` (P = derived) back into the field.

## Determinism (the whole point — tol 0, no re-baseline)

- The sim path is integer **Q16.16 / Q.32** (iron rule: no floats). Integer add/mul are
  associative → GPU parallel reductions CAN be bit-exact if you use deterministic reduction trees
  or integer atomics; **verify each reduction matches the CPU fold value, not just "close."**
- The **Galerkin coarsening gathers** are the trap: the on-device build must reproduce the CPU's
  exact operator entries and fold order. `cuda_mg_solve.h` §1.1/§1.2 has the determinism argument —
  read it before designing the gather kernels.
- Warm start (P_prev) and the Q.32 reciprocals must be bit-for-bit — a reciprocal off by 1 LSB
  propagates through the whole V-cycle.

## The gate (extend Path B's)

Path B ships `tests/cuda_s8a_check.py` (live A/B, residency-ON vs CPU, tol 0 — self-referential, no
stored golden to reproduce; "no re-baseline" applies to the per-kernel digest baselines the existing
CUDA gates assert). For Path A: the SAME 30-tick full-engine A/B must stay tol-0 with EOS now
resident (bracket removed), plus the existing `tests/cuda_eos_step_check.py` per-kernel digest gate
must stay green. PART 2 benchmark should now show the EOS transfer tax gone too (big-map win).

## Suggested approach (your call — design-pass it)

Port in determinism-safe order, each piece gated before the next: (1) host reductions → device
(easy, integer, order-independent); (2) `mg_vcycle`/`kick_compression` device-pointer rework (they're
already kernels — mostly plumbing); (3) `mg_build_levels` on device (the hard gather port — do a
design doc + critique for THIS piece specifically); (4) remove the D2H/H2D bracket; (5) gate.
Keep the per-call `eos_step_cuda` path working as the live fallback throughout.

## Reference files
`cpp/src/cuda_eos_step.cu` (the island) · `cuda_mg_solve.cu`/`.h` (vcycle + the §1/§2.7 determinism
notes) · `cuda_kick_compression.cu`/`.h` · `cuda_sl_advection.*` · `eos_solver.cpp`
(`mg_build_levels` + the host reductions source-of-truth) · `eos_p6_gpu_alignment_review.md`
(the pass-boundary map + cost model). Build: `cpp/build_cuda_lenovo.bat`; env python
`C:/Users/steen/miniconda3/envs/data/python.exe` (NOT `conda run` — plugin crash on this box).
