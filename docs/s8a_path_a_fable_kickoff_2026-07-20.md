# S8a Path A — Fable kickoff prompt (paste this at midnight)

> Ready-to-paste session prompt for the Fable model to build S8a Path A (EOS device
> residency). Set the model to Fable, paste the block below. The technical detail
> lives in `docs/s8a_path_a_eos_residency_brief_2026-07-20.md`; this is the wrapper
> that orients + sets the workflow.

---

You are **Fable**, closing out **S8a — Path A: EOS device residency** on the breach project.
You wrote the S8a residency plan; today Opus built **Path B** (leaf-solver residency — water/smoke
resident, EOS bracketed) and merged it to main (`1ae6f86`). Your job is the determinism-critical
other half: make the **EOS pressure stage** device-resident so the synced fields never leave the GPU
across a whole tick, removing the D2H/H2D bracket Path B left around EOS.

**Work the autonomous-patch-workflow model — DESIGN PASS FIRST, then build.** This is the single most
determinism-fragile part of the engine (one wrong index or non-deterministic reduction silently
breaks bit-identity). Do NOT jump to code:
1. Read `docs/s8a_path_a_eos_residency_brief_2026-07-20.md` (the EOS host island, what to port, the
   determinism traps, the gate) + the ★ BUILD FINDING block in
   `docs/cuda_s8a_residency_spec_2026-07-19.md` + `cuda_mg_solve.h` §1.1/§1.2/§2.7.
2. Write a `PathA_impl.md` design doc: how you port `mg_build_levels` (the Galerkin coarse operators,
   Q.32 diagonal reciprocals, P_prev warm start) + the host reductions (`div_u`, Dalton/`pstar`,
   `c_local`/substep-count) to device, bit-exact; how `eos_mg_vcycle`/`eos_kick_compression` take
   device pointers from the resident set; where you drop the bracket.
3. Run an **independent adversarial critique** across distinct lenses — (a) bit-exactness of the
   Galerkin gathers vs the CPU operator entries + fold order, (b) integer-reduction determinism
   (Q16.16/Q.32 is associative, but the reduction must match the CPU value, not just "close"),
   (c) the vcycle/kick device-pointer rework + scope/regression. Resolve blockers on paper, iterate
   the doc until it survives, THEN build.

**The build (into Path B's framework):** the EOS bracket you remove is step 4 of
`PhysicsRunner._step_resident` — `run_substeps(..., do_traces=False)` running on the host mirror.
Replace it with device-pointer EOS kernels reading the already-resident `atmosphere`/`wave_p`/
`wind_x/y`/`temperature`/`gas` (already CuPy in `_dev`; `device_ptrs()` exposes them), and drop the
surrounding `from_host`/`to_host`. Keep the per-call `eos_step_cuda` path working as the live fallback.

**Environment (Lenovo/Ada):** Python = `C:/Users/steen/miniconda3/envs/data/python.exe` (NOT
`conda run -n data` — conda-plugin crash on this box). CUDA build = `cpp/build_cuda_lenovo.bat`.
⚠ **After any merge, rebuild the MAIN tree's `.pyd` before running `main.py` there** — worktree
builds don't propagate (Opus hit this today: a stale binary crashed `--cuda --resident`).

**Gate (the oracle):** extend `tests/cuda_s8a_check.py` — the ≥30-tick full-engine A/B stays **tol 0**
with EOS now resident (bracket gone), incl host-path heat/ripple; the per-kernel digest gates
(`cuda_eos_step_check.py`) stay green. **NO re-baseline.** Full `pytest tests -q` green. PART 2 should
now show the EOS transfer tax gone too (the big-map win Erik wants — transfer is his measured
bottleneck).

**Discipline:** bit-identical tol 0, no physics change, in-place-only resident fields (the
`__setattr__` guard), CPU + per-call GPU stay the live defaults, residency opt-in behind the flag
(default OFF). **Auto-merge on green** is authorized ONLY after the design pass clears AND the gate +
full suite are green. If the `mg_build_levels` port balloons or the gate won't go tol-0: **STOP,
push the branch building-but-not-merged, report exactly what's done vs blocking. Never merge red.**
Never `git add -A`; don't touch untracked `levels/test_level/*.bak`. End commit messages with
`Co-Authored-By: <the Fable model's credit line>`.

**After Path A lands, S8a is complete** → physics-v1 closes → next is S8b (CUDA graphs), then S8c
(render CUDA-GL interop + recorder kernels + the `cast_fire_heat` device port — where the fire-FPS
fix lives). Those are NOT this task.

---
