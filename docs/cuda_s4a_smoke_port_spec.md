# CUDA-S4a — SmokeDynamics::step GPU port spec

**Status:** in progress (branch `cuda-s4a-smoke`).
**Goal:** a faithful, **bit-identical** GPU port of `SmokeDynamics::step`
(`cpp/src/smoke_dynamics.cpp`) — per-gas smoke transport: diffusion (Laplacian) →
semi-Lagrangian wind advection → clamp/zero. Multi-gas. The synced `gas` field
(int32 Q16.16, shape `(N_gases, h, w)`) must come out **byte-for-byte identical**
CPU vs GPU (tol 0) on **every plane**.

**NOT in scope / DO NOT TOUCH:**
- `SmokeDynamics::sink_hop` (the breach-pull) — that's **S4b** (it reuses this
  advection machinery; porting it separately keeps the patch small). Leave it on
  the CPU; with the smoke backend on, `step` runs GPU and `sink_hop` runs CPU in
  BOTH the reference and the test path, so the integration gate stays valid.
- No smoke-physics change of any kind. The gate enforces this.
- `docs/architecture/engine` smoke chapter — comment/doc only if needed, no model change.

Mirror the S1/S3 template: a `.cu` of grid-stride gather kernels + a plain-C++ `.h`
+ a backend flag (`smoke_backend_is_cuda`/`set_smoke_backend_cuda`) + a `PhysicsEngine`
dispatch + an isolated pybind gate binding + a subprocess pytest. Read the S3 files
(`cuda_water.{h,cu}`, the bindings/dispatch) as the working template.

---

## 0. PREREQUISITE — share the device 128-bit helper (do this FIRST)

S3 put `mul128_shr_signed` + `recip_mul_dev` as file-locals in `cuda_water.cu`.
Extract them into a NEW shared device header **`cpp/src/cuda_fixedpoint_device.cuh`**
so S4 (and S7 atmosphere) reuse them instead of copy-pasting:
- `__device__ int64_t mul128_shr_signed(int64_t a, int64_t b, int S)` — verbatim from
  cuda_water.cu (the `__mul64hi` + `(lo>>S)|(hi<<(64-S))` combine; bit-identical to the
  host MSVC `_mul128` path; see [[fixed_point_migration_lessons]] #10).
- `__device__ q16 recip_mul_dev(q16 x, int64_t recip)` — verbatim (S=`RECIP_SHIFT`).
- `__device__ q16 reciprocal_q16_dev(q16 denom_q)` — **NEW**: a device port of
  `fixedpoint::reciprocal_q16` (fixed_point.h:255), the Newton-iteration reciprocal
  the smoke bilinear renorm uses. Replicate the host integer math EXACTLY, swapping
  any host `_mul128`/`recip_mul` 128-bit step for `mul128_shr_signed`. (reciprocal_q16
  is also S7-atmosphere's GS denominator — sharing it now pays twice.)
- **Why a device helper, not `fixedpoint::recip_mul`:** MSVC-host nvcc has no device
  `__int128`; the header `recip_mul`/`reciprocal_q16` device instantiation would resolve
  to the host-only `_mul128` branch and fail to compile if ODR-used on device.

Then in `cuda_water.cu`: delete the two local copies, `#include "cuda_fixedpoint_device.cuh"`.
**Re-run the S3 gate** (`tests/cuda_s3_check.py` via the harness) to PROVE the refactor
is bit-identical (tol 0) — it must still PASS before you build any smoke code.

---

## 1. The passes (mirror `SmokeDynamics::step` — READ smoke_dynamics.cpp ~187-302)

Scout map (verify line numbers against the actual file):

| # | pass | per | reads | writes | notes |
|---|------|-----|-------|--------|-------|
| K1 | diffusion Laplacian (~221-238) | cell | smoke (4-nbr), permeability (per-face float bridge) | `lap[]` scratch | `neighbor_q = quantize((double)perm_face)` per face; integer 4-stencil. |
| K2 | diffusion apply (~240-256) | cell | wind_x/y, lap[] | smoke (in-place) | `d_eff = d_smoke·(1 + wind_diffusion_scale·\|wind\|²)` in DOUBLE; `wind_sq` via `mul_wide`(Q.32)→dequantize once→fold→`quantize(d_eff·dt)` once per cell; `smoke += d_eff_dt·lap`. **--fmad=false.** |
| K3 | **semi-Lagrangian advection** (~267-289) — THE HARD KERNEL | cell | a snapshot of smoke (replicate the CPU's EXACT snapshot timing into `src_`), wind | smoke (in-place) | displacement `bx_q,by_q = −mul_q16(wind[i], dt_adv_q)` (pure integer; `dt_adv_q = quantize(advection_rate·dt)` once). **DDA wall-clip march** (sqrt-free, dominant-axis/Chebyshev). Integer **bilinear** at the back-traced point with **`reciprocal_q16_dev` renorm** (WSUM floor=`WSUM_FLOOR_Q`=256, eps=`WSUM_EPS_Q`=4). |
| K4 | clamp + zero (~292-299) | cell | smoke, masks | smoke (in-place) | clamp `[0, SMOKE_MAX_Q]`; zero on wall/vacuum. |

Host-side scalar precompute (in the entry fn, double, verbatim from the CPU top-of-step):
`dt_adv_q`, the diffusion scalars, any `quantize`d constants. `d_smoke` (per-gas) is a
scalar arg. Device scratch: `lap[]`, `src_[]` (the advection snapshot). Kernels separate
launches (barriers). Every thread writes its own cell fully (no uninitialised scratch).

**The advection determinism contract (where bugs hide):**
1. **Exact DDA loop** — same structure, no unrolling/reordering of the march; same
   early-break conditions (wall/breach clip).
2. **Negative-displacement rounding** — the per-step increment / cell-index derivation
   must match the CPU's integer floor-divide / `>>` semantics for NEGATIVE displacements
   bit-for-bit (this is the #1 risk — see [[project_cuda_migration]] G-note).
3. **Bilinear accumulation in int64**, same corner order (associative int64 sum is
   order-safe, but keep the order anyway), `reciprocal_q16_dev` for the renorm.
4. **`--fmad=false`** (already applied to all CUDA TUs) for the double `d_eff` fold and
   any float bridge.

---

## 2. Multi-gas dispatch (physics_engine.cpp ~267-331)

The smoke runs nested: `for s in n_smoke (substeps): for gi in n_gases: if any(slice): smoke.step(gas + gi·plane, ...)` with `this->smoke.d_smoke = gas_diffusion[gi]` set per gas. Port as a **kernel-per-slice host loop** (residency deferred to S8): for each substep × gas slice, if the host-side `any()`-nonzero check passes, launch the K1-K4 chain on `gas + gi·plane` with `d_smoke = gas_diffusion[gi]` (a scalar arg). Per-call H2D/D2H of the plane (S1/S3 pattern; note the transfer cost in a comment — residency is S8). Wrap the inner `smoke.step` call site in `#ifdef BREACH_HAS_CUDA / if (smoke_backend_is_cuda()) { breach_cuda::smoke_step(...) } else #endif { this->smoke.step(...) }`. Keep the `any()`-skip on the host (N_gases small).

---

## 3. Files

- `cpp/src/cuda_fixedpoint_device.cuh` (NEW) — the shared device helpers (§0).
- `cpp/src/cuda_water.cu` — drop local helpers, include the shared header; **re-gate S3**.
- `cpp/src/cuda_smoke.{h,cu}` (NEW) — the 4-kernel port + host entry + backend flag.
- `cpp/CMakeLists.txt` — add `src/cuda_smoke.cu` to the BREACH_CUDA list.
- `cpp/src/physics_engine.cpp` — the multi-gas GPU dispatch (§2); include `cuda_smoke.h`.
- `cpp/src/bindings.cpp` — `set/get_smoke_backend` + a `cuda_smoke_step(...)` isolated
  gate binding (nullable perm/wind handled like the live `SmokeDynamics.step` binding).
- `tests/cuda_s4a_check.py` (NEW, mirror cuda_s3_check.py):
  - **PART 1 isolated:** synthetic gas + wind (BOTH signs, high magnitude → multi-cell +
    **NEGATIVE-displacement** advection + wall-clip), permeability variation, wall/vacuum
    masks, several gas planes, degenerate 1×N/N×1, many seeds. `SmokeDynamics().step` vs
    `cuda_smoke_step` on identical copies; `np.array_equal` tol 0 per plane. MUST hit:
    negative-displacement advection, the diffusion wind² term, the permeability bridge,
    vacuum/wall zeroing, and a `reciprocal_q16` renorm with WSUM near the floor.
  - **PART 2 integration:** a seeded smoke+wind scenario through both `PhysicsEngine` smoke
    backends via `set_smoke_backend()`; full per-tick `gas` trajectory bit-identical over
    30 ticks; default-scenario CPU digest still `60bd331f…`. Print `S4A_RESULT: PASS`/`FAIL`.
- `tests/test_cuda_s4a_smoke.py` (NEW) — pytest wrapper (skipif no CUDA, subprocess harness).

---

## 4. Build + gate

Same as S3: CUDA build `cpp/build_cuda.bat`; CPU build `cpp/build/Release`; interpreter
`C:/Users/steen/anaconda3/python.exe`; gate `cuda_s4a_check.py` → `S4A_RESULT: PASS`; full
suite `-m pytest tests/ --ignore=tests/test_main_smoke.py --ignore=tests/test_renderer_smoke.py`.
The bit-identity gate (tol 0) IS the oracle → auto-merge on green. **Also re-run the S3 gate**
after the §0 refactor.

**Top risks:** (1) DDA negative-displacement floor-divide; (2) `reciprocal_q16` device port;
(3) wind² double-fold FMA; (4) bilinear corner order. See [[project_cuda_migration]] tail +
the S4 scout. The advection is the hardest kernel of the arc so far — verify the negative
case explicitly (revert-the-fix style: a deliberately-wrong floor-divide should FAIL the gate).
