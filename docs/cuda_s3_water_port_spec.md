# CUDA-S3 — WaterSolver GPU port spec

**Status:** in progress (branch `cuda-s3-water`).
**Goal:** a faithful, **bit-identical** GPU port of `WaterSolver::step`
(`cpp/src/water_solver.cpp`) — the pipe model: surface potential → damped velocity
kick → donor-cell upwind flux → outflow limiter → divergence apply → clamps. The
three synced fields `water_depth`, `flow_vx`, `flow_vy` (int32 Q16.16) must come
out **byte-for-byte identical** CPU vs GPU (tol 0), the whole point of the arc.

**NOT in scope / DO NOT TOUCH:**
- `WaterSolver::step_ripple` — render-only FLOAT, stays on the CPU. Not ported.
- `docs/architecture/engine/07_fluid_and_water.md` — canon water chapter (Fable's
  domain). This port changes **zero** water behaviour; do not edit ch.07.
- No water-physics change of any kind. The gate enforces this.

This mirrors the S1 template (`cuda_temperature.{h,cu}`) one-to-one: a gather,
multi-pass, integer kernel + a backend flag + a `PhysicsEngine` dispatch + an
isolated pybind gate binding + a subprocess-isolated pytest.

---

## 1. The determinism contract (the careful bits)

`water_solver.cpp` is already compiled `/fp:strict` (CMakeLists:126) and is on the
sim-state determinism floor. The CPU reference is already deterministic. The port
must keep it bit-identical on the GPU. The only non-trivially-integer ops:

1. **Host scalar precompute.** Replicate water_solver.cpp lines 54–79 **on the host**
   inside the `.cu` entry function, in `double`, exactly as the CPU does:
   `g_dt_q`, `damp_dt_q`, `v_max_q`, `recip_two_dx = make_recip(2*dx)`,
   `dt_over_dx_q`, `depth_eps_q`, the TILT clamp + `tan_tx = tan_poly(quantize(txd))`,
   `tan_ty`, `cx = 0.5*w`, `cy = 0.5*h`, `dx_d`, `head_on = (k_p!=0)`, `kp_f`.
   Pass these as **scalar kernel args**. This guarantees the scalar precompute is
   identical (it is literally the same host code). `make_recip` therefore stays
   host-only — no promotion needed.

2. **Per-tile tilt — DOUBLE on device.** Only `quantize(((double)x - cx) * dx_d)`
   and `quantize(((double)y - cy) * dx_d)` are per-tile, so they run on the device
   in `double`. With `--fmad=false` (already set for all CUDA TUs, CMakeLists:86)
   double sub/mul/`quantize` do not contract, so they are bit-identical to the CPU
   `/fp:strict` path. Use the `FP_HD quantize(double)` (fixed_point.h:88). Each
   thread recomputes its own `tilt_row + tilt_col` (the CPU hoists `tilt_row` to the
   row loop — pure optimization, identical values).

3. **`recip_mul` → promote to `FP_HD`.** The central-difference gradient
   (`recip_mul((q16)(s_e - s_w), recip_two_dx)`) needs the device. Add `FP_HD` to
   the single `recip_mul` definition in fixed_point.h. On the device nvcc defines
   `__SIZEOF_INT128__` → the `__int128` branch compiles; the `_MSC_VER`/`_mul128`
   branch is host-only and never reached on device. The CPU build keeps `_mul128`.
   These were engineered bit-identical (header note 162–165). **Verify at build**
   that nvcc takes the `__int128` device branch (if it somehow doesn't define
   `__SIZEOF_INT128__`, write a device-local `__int128` recip_mul instead).
   This is the sanctioned G3/G4 promotion the header comment (49) anticipated.

4. **`flux_to_dq` — device `__int128`.** The CPU lambda (water_solver.cpp:208–230)
   uses `_mul128` on MSVC, proven bit-identical to the `__int128` path
   (`tests/_s1_flux_truncation_check.py`). Write a `__device__` helper:
   `q16 flux_to_dq_dev(int64_t flux_wide, q16 dt_over_dx_q){ __int128 p =
   (__int128)flux_wide * dt_over_dx_q; return (q16)(p >> 32); }`.

5. **Head FLOAT BRIDGE (`k_p != 0`).** Default config ships `k_p = 0.5` (head LIVE
   every substep). When `head_on`, the kernel reads the **host-dequantized float**
   `atmosphere`/`wave_p` bridge arrays (passed in as `const float*`, nullable →
   skip), forms `head_f = kp_f*(atm + wp)` in **float**, then `quantize((double)
   head_f)`. `--fmad=false` keeps the float `mul(add())` from fusing → bit-identical.
   Pass `kp_f` + the two float arrays + `head_on` to the surface kernel.

6. **Outflow limiter** is an exact int64 divide `(depth<<16)/out_sum` → deterministic
   on device. **`scale_mag`** (already FP_HD) truncates on MAGNITUDE — use it, NOT
   `mul_q16`, exactly as the CPU (the `>>16`-toward-−∞ of `mul_q16` would over-drain
   a negative outgoing delta and break conservation).

---

## 2. Kernel decomposition (8 passes, each a launch = barrier)

Mirror the CPU passes; data deps make each pass a separate kernel launch sharing
device buffers. Scratch device buffers: `d_surface (q16)`, `d_fx/d_fy (int64)`,
`d_dq_e/d_dq_s (q16)`, `d_scale (q16)`. Inputs: `d_depth/d_vx/d_vy (q16, in/out)`,
`d_floor (q16, nullable→pass flat-zero or a null flag)`, `d_solid (bool)`,
`d_atm_f/d_wave_f (float, nullable, head bridge)`.

| # | kernel | per | reads (frozen) | writes |
|---|--------|-----|----------------|--------|
| K1 | `water_surface`   | cell | depth, floor, (atm_f,wave_f if head_on) | surface |
| K2 | `water_velocity`  | cell | surface (Neumann mirror), own vx/vy | vx, vy (0 on solid) |
| K3 | `water_flux`      | cell | vx,vy (updated), depth | fx[i],fy[i] (0 on solid/border) |
| K4 | `water_dq`        | cell | fx,fy | dq_e[i],dq_s[i] (flux_to_dq, 0 if flux 0) |
| K5 | `water_scale`     | cell | dq_e[i],dq_e[i-1],dq_s[i],dq_s[i-w], depth | scale[i] (FP_ONE default) |
| K6 | `water_scale_app` | cell | scale (frozen), dq_e[i]/dq_s[i] | dq_e[i],dq_s[i] (scale_mag) |
| K7 | `water_diverge`   | cell | dq_e[i],dq_e[i-1],dq_s[i],dq_s[i-w] | depth (−= div) |
| K8 | `water_clamp`     | cell | depth, solid | depth (max0, 0 on solid, eps snap) |

Every thread writes its own cell fully (no uninitialised scratch read). Border/solid
faces write 0. `block=256`, grid = ceil(n/256), grid-stride loop (S1 style).

Host entry `breach_cuda::water_step(...)`: malloc + H2D + host scalar precompute +
8 launches + `cudaDeviceSynchronize` + D2H (depth, vx, vy only) + free. Per-substep
call (n substeps → n entry calls, H2D/D2H each — the S1 per-call pattern; GPU
residency across substeps is deferred to S8).

---

## 3. Files to create / change

- **`cpp/src/cuda_water.h`** (NEW, plain C++ header, no CUDA types — mirror
  cuda_temperature.h): declare `void water_step(q16* water_depth, q16* flow_vx,
  q16* flow_vy, const q16* floor_height, const float* atmosphere, const float*
  wave_p, const bool* solid, int h, int w, float dt, float tilt_x, float tilt_y,
  float g, float damping, float dx, float k_p, float v_max, float depth_eps)` +
  `bool water_backend_is_cuda(); void set_water_backend_cuda(bool);`. (Use `int32_t`
  in place of `q16` in the header so the `.cpp` TUs need no fixed_point include, OR
  include fixed_point.h — match whichever cuda_temperature.h does; it uses `int32_t`.)
- **`cpp/src/cuda_water.cu`** (NEW): the 8 kernels + `flux_to_dq_dev` + host entry +
  backend flag, per §1–§2. Include `fixed_point.h` for the FP_HD helpers.
- **`cpp/src/fixed_point.h`**: add `FP_HD` to `recip_mul` (the one promotion). Update
  the 46–49 comment to record that S3 water promoted it.
- **`cpp/CMakeLists.txt`**: add `src/cuda_water.cu` to the `BREACH_CUDA` source list
  (after `src/cuda_raycaster.cu`). No `/fp:strict` change (water already there).
- **`cpp/src/physics_engine.cpp`** (`step_water`, ~line 399): wrap the per-substep
  `this->water.step(...)` in `#ifdef BREACH_HAS_CUDA / if (water_backend_is_cuda())
  { breach_cuda::water_step(... this->water.g, damping, dx, k_p, v_max, depth_eps);
  } else #endif { this->water.step(...); }`. CPU path stays the live fallback.
  Include `cuda_water.h` (guarded like cuda_temperature.h is).
- **`cpp/src/bindings.cpp`**: add `set_water_backend` / `get_water_backend` +
  `cuda_water_step(...)` isolated gate binding (mirror `cuda_temperature_step`,
  76–95; nullable floor/atmosphere/wave_p like the live `WaterSolver.step` binding
  595–633). All under `#ifdef BREACH_HAS_CUDA`.
- **`tests/cuda_s3_check.py`** (NEW, mirror `cuda_s1_check.py`):
  - **PART 1 ISOLATED:** synthetic inputs hitting every branch — random depth (incl.
    0 + large), vx/vy both signs near ±v_max, random solid mask, floor (null + explicit),
    nonzero tilt_x/tilt_y (tilt poly + double tilt product), `k_p ∈ {0, 0.5}` with
    random float atm/wave_p (head bridge), **convergent high-velocity + shallow-depth
    patches to FORCE the outflow limiter** (`out_sum > depth`), several grid sizes incl.
    degenerate 1×N / N×1, many seeds. CPU `WaterSolver().step` vs `cuda_water_step` on
    identical copies; assert `np.array_equal` on depth AND vx AND vy (tol 0).
  - **PART 2 INTEGRATION:** a seeded-water A/B scenario (a `make_wet()` like S1's
    `make_hot()` — seed a water blob + a tilt so transport actually evolves) under both
    `PhysicsEngine` backends via `set_water_backend()`; assert the full per-tick synced
    trajectory is bit-identical over N ticks; confirm the default-scenario CPU digest
    still equals the golden (`60bd331f…` — water backend off changes nothing). Print
    `S3_RESULT: PASS`/`FAIL`, exit 0/1.
- **`tests/test_cuda_s3_water.py`** (NEW): pytest wrapper mirroring
  `test_cuda_s1_temperature.py` — `skipif` no CUDA, subprocess via
  `cuda_harness.run_cuda_script`.

---

## 4. Build + gate

- CPU build (unchanged): `cpp/build/Release` via the anaconda cmake.
- CUDA build: `cpp/build_cuda.bat` (vcvars64 + Ninja + nvcc, archs 75/86/89, RTX 3070).
- Run gate: `C:/Users/steen/anaconda3/python.exe tests/cuda_s3_check.py` inside the
  CUDA build env (it imports the CUDA `breach_physics` first, S1 style), expect
  `S3_RESULT: PASS`. Then the full suite:
  `… -m pytest tests/ --ignore=tests/test_main_smoke.py --ignore=tests/test_renderer_smoke.py`.
- The bit-identity gate (tol 0) IS the correctness oracle → auto-merge on green.
