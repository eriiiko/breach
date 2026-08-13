# CUDA-S7 — AtmosphereSolver::diffuse_solve GPU port spec (the LAST + hardest solver)

**Status:** in progress (branch `cuda-s7-diffuse`).
**Goal:** a faithful, **bit-identical** GPU port of `AtmosphereSolver::diffuse_solve`
(`cpp/src/atmosphere_solver.cpp` ~269-601) — the once-per-tick implicit atmosphere step:
RHS snapshot → Red-Black Gauss-Seidel pressure relaxation → vacuum BFS/sponge → wind
gradient. The synced fields **atmosphere, wave_p, wave_v, wave_source, wind_x, wind_y**
(int32 Q16.16) must come out **byte-for-byte identical** CPU vs GPU (tol 0).

**SCOPE:** the WHOLE `diffuse_solve` as ONE patch (one `diffuse_solve_gpu` entry + one
backend flag `atmos_backend_is_cuda`/`set_atmos_backend_cuda` + one gate). With S5 (wave)
+ this, the atmosphere/wave system is fully GPU. `wave_solver.cpp` is dead (don't touch).
No physics change. This is the deliberately-last, hardest kernel (the RB-GS).

Mirror the S1-S6 template + the shared `cuda_fixedpoint_device.cuh`.

---

## 1. The passes (mirror `diffuse_solve` — READ atmosphere_solver.cpp ~269-601)

Host scalar precompute (in the entry fn, double, verbatim from the CPU top ~290-300):
`dt = sim_time`, `mu_q = quantize(d_atm * sim_time)`, `gs_iters` (member, =8), the sponge
constants `atm_vac_k_q`/`atm_inner_k_q`/`atm_outer_k_q`/`wv_inner_k_q`/`wv_outer_k_q`/
`ws_half_q` (all `quantize(1-η·…)` once), `MU_EPS_Q`. Pass as scalar kernel args.

| # | pass | shape | notes |
|---|------|-------|-------|
| GATE | μ-gate (~309) | host | `if (mu_q > MU_EPS_Q)` → run RHS + GS; ELSE skip them (atmosphere unchanged by GS). **The sponge + wind ALWAYS run** regardless. |
| K0 | Dinv + RHS (~310-368) | per-cell | `rhs[i]=atmosphere[i]`; per-cell `wsum = Σ4 face_q` (face = `quantize((double)min(perm_i,perm[nb]))`, 0 if OOB/obstacle/wall); `denom = FP_ONE + mul_q16(mu_q,wsum)`; `dinv[i] = reciprocal_q16_dev(denom)`. **RECOMPUTE unconditionally** (drop the CPU `dinv_key_` cache). Sentinel `dinv[i]=0` on obstacle/wall/vacuum. Device scratch arrays `d_rhs`, `d_dinv`. |
| K_GS | Red-Black GS (~372-414) | per-colour | **the CRUX.** `for iter in gs_iters(8): for color in {0,1}: launch a kernel over (x+y)&1==color cells.** Each cell (skip obstacle/wall/vacuum): `ai=atm[i]`; `acc=Σ4 mul_wide(muw, atm[nb]-ai)` where `muw=mul_q16(mu_q, face_q)` (0 if OOB/obstacle/wall); `flux=narrow(acc)`; `resi = flux - (ai - rhs[i])`; `inc = round_nearest_q_dev((int64)resi*(int64)dinv[i])`; `atm[i]=ai+inc`. **TWO launches per iter (red then black)** = the colour-schedule barrier (RED reads only BLACK). 16 launches total. round_nearest (sign-symmetric), NOT mul_q16 toward −∞. At the fixed point inc→0 (drift-free). |
| K_BFS1 | vacuum dist=1 (~483-495) | gather | seed `vac_dist=0` where `is_vacuum && !obstacle && !is_wall` (a pre-pass or host); K_BFS1: a non-seed non-solid cell → `dist=1` if any 4-neighbour `dist==0`. **Reads ONLY the frozen seed level → order-free gather.** Double-buffer (read in / write out) for obvious race-freedom. |
| K_BFS2 | vacuum dist=2 (~497-509) | gather | `dist=2` if `dist==255` and any neighbour `dist==1`. Reads only the frozen dist=1 level → order-free. Separate launch (barrier after K_BFS1). |
| K_SPONGE | sponge relax (~511-548) | per-cell | per `vac_dist[i]`: 0→`atm=mul_q16(atm,atm_vac_k_q); wave_p=0; wave_v=0`; solid(obstacle/wall)→`wave_p=wave_v=atm=0`; 1→`atm=mul_q16(atm,atm_inner_k_q); wave_v=scale_mag_dev(wave_v,wv_inner_k_q); wave_source=0`; 2→`atm=mul_q16(atm,atm_outer_k_q); wave_v=scale_mag_dev(wave_v,wv_outer_k_q); wave_source=mul_q16(wave_source,ws_half_q)`. Per-cell → order-free. |
| K_WIND | wind gradient (~550-595) | per-cell | skip→`wind=0` on obstacle/wall/vacuum. `p_total(idx)=atmosphere[idx]+wave_p[idx]` (both int32, no dequantize); per-face `f=quantize((double)min(perm_i,perm[nb]))` (0 if OOB); `p_side = p_here + mul_q16(f, p_total(nb)-p_here)`; `wind_x[i] = -shr_round0_dev(p_right-p_left, 1)`; `wind_y[i] = -shr_round0_dev(p_down-p_up, 1)`. |

**Pass order (separate launches = barriers):** [μ-gate→] K0 → (K_GS_red, K_GS_black)×8 → K_BFS1 → K_BFS2 → K_SPONGE → K_WIND. The sponge reads the post-GS atmosphere + the wave fields; the wind reads the post-sponge atmosphere + wave_p. Preserve this order exactly.

**`last_gs_residual`** is a host-side FLOAT diagnostic (Linf residual norm), **NOT in the
digest** (not synced). CHECK whether any test/caller reads it: if not, the GPU entry may
skip it (leave the member as the CPU last set, or 0). If a test does read it, compute the
integer intermediates (`res_max_q`, `atm_absmax_q`) via order-free int reductions and the
final float dequantize on host (matching the CPU) — but it is NOT gated tol-0 (float).

---

## 2. The one new device helper

Add to `cpp/src/cuda_fixedpoint_device.cuh` (additions-only, after `sqrt_q16_dev`):
```cuda
__device__ __forceinline__ q16 shr_round0_dev(q16 x, int s) {   // sign-symmetric >>
    return (x < 0) ? -((-x) >> s) : (x >> s);
}
```
(verbatim of `fixedpoint::shr_round0`, fixed_point.h:213-215). All else exists:
`reciprocal_q16_dev` (Dinv), `round_nearest_q_dev` (GS increment), `scale_mag_dev`
(sponge wave_v), plus FP_HD `mul_q16`/`mul_wide`/`narrow`/`quantize`. **No 128-bit
needed** (resi·dinv fits int64).

---

## 3. Files
- `cpp/src/cuda_fixedpoint_device.cuh` — add `shr_round0_dev`.
- `cpp/src/cuda_atmosphere.{h,cu}` (NEW): the kernels + the host entry `diffuse_solve_gpu(...)`
  (host scalar precompute, the μ-gate, K0→GS×16→BFS→sponge→wind, device scratch d_rhs/d_dinv/
  d_vacdist[×2 double-buffer], per-call H2D/D2H; residency S8) + backend flag. `#include
  "cuda_fixedpoint_device.cuh"`. (Name it `cuda_atmosphere` since it's the atmosphere solver;
  if a `cuda_atmosphere.*` already exists from S2c, extend it — CHECK first.)
- `cpp/CMakeLists.txt` — add the new `.cu` to BREACH_CUDA.
- `cpp/src/physics_engine.cpp` (~275-281): wrap the `atmos.diffuse_solve(...)` call in
  `#ifdef BREACH_HAS_CUDA / if (breach_cuda::atmos_backend_is_cuda()) { breach_cuda::
  diffuse_solve_gpu(...) } else #endif { ... }`. Pass the solver's scalar dials explicitly.
  Include the new header guarded.
- `cpp/src/bindings.cpp` — `set/get_atmos_backend` + a `cuda_diffuse_solve(...)` isolated
  gate binding (mirror the live `AtmosphereSolver.diffuse_solve` binding; pass scalars).
- `tests/cuda_s7_check.py` (NEW, mirror cuda_s5_check.py):
  - **PART 1 isolated:** synthetic atmosphere/wave_p/wave_v/wave_source + perm + masks +
    **vacuum patterns (exposed-vacuum seeds → exercise BFS dist 0/1/2/255)** + ± wave_v
    (sponge scale_mag) + varied μ (incl. μ≤MU_EPS → GS-skip path, and large μ → strong
    diffusion). Run `diffuse_solve` CPU vs `cuda_diffuse_solve` on identical copies;
    `np.array_equal` tol 0 on ALL SIX fields (atmosphere, wave_p, wave_v, wave_source,
    wind_x, wind_y). MUST hit: the GS convergence (multi-iter), the μ-gate skip, the BFS
    layers, the sponge tiers, the wind gradient (± gradients), degenerate grids, many seeds.
  - **PART 2 integration:** a scenario with pressure gradients + a breach (so the GS, sponge,
    AND wind are all exercised) through both `PhysicsEngine` atmos backends via
    `set_atmos_backend()`; full per-tick trajectory of the 6 fields bit-identical over 30
    ticks; default-scenario CPU digest still `60bd331f…`. Print `S7_RESULT: PASS`/`FAIL`.
- `tests/test_cuda_s7_diffuse.py` (NEW) — pytest wrapper.

---

## 4. Build + gate
Same as S5/S6. Bit-identity tol 0 on the 6 fields IS the oracle → auto-merge on green.
**Top risks (this is the hardest stage):** (1) the GS residual-form increment rounding
(`round_nearest_q_dev`, sign-symmetric — a toward-−∞ slip = DC mass drift); (2) the
two-launch red-black schedule (RED kernel must read only the frozen BLACK + write only RED);
(3) the BFS two-pass double-buffering (each pass reads only the prior frozen level —
order-free, but double-buffer to be obviously race-free); (4) the Dinv unconditional
recompute (drop the cache); (5) the perm bridge in GS + wind (device `quantize`, `--fmad=false`).
The gate must converge the GS over multiple iters AND exercise μ≤eps (skip) + the BFS + the
sponge tiers. Prove the GS is drift-free (a uniform field stays uniform).

---

**Appended 2026-08-14 (supersession note).** Any ×2 game-T→Kelvin map referenced
above is superseded by the unified canonical map in
`[physics.temperature_scale]` (`K = 293 + 3·T_game`; EOS pressure calibration
keeps a named, deliberate exception at `eos_t_amb_k = 290`). See
`docs/temperature_scale_unification_design_2026-08-13.md`.
