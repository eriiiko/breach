# CUDA-S5 — AtmosphereSolver::wave_substep GPU port spec

**Status:** in progress (branch `cuda-s5-wave`).
**Goal:** a faithful, **bit-identical** GPU port of `AtmosphereSolver::wave_substep`
(`cpp/src/atmosphere_solver.cpp` ~62-261) — the explicit damped-wave shockwave step:
source feed → Laplacian → velocity kick → pressure update → absorption → wave BCs →
**mean_wp global reduction** → anomaly transfer to atmosphere. The synced fields
`wave_p`, `wave_v`, `wave_source` (int32 Q16.16) AND `atmosphere` (modified by the
anomaly transfer) must come out **byte-for-byte identical** CPU vs GPU (tol 0).

**SCOPE DECISION — read carefully:**
- **IN scope:** `wave_substep` ONLY (the n_wave-loop function).
- **NOT in scope:** `diffuse_solve` (the once-per-tick implicit step: RHS snapshot +
  Red-Black Gauss-Seidel + vacuum BFS/sponge + the **WIND gradient**). That whole
  function = **S7** (the RB-GS is the deliberately-last, hardest kernel; the wind lives
  inside it and reads the post-GS atmosphere). So **S5 does NOT produce wind_x/wind_y**
  and does NOT touch the GS. With the wave backend on, `wave_substep` runs GPU and
  `diffuse_solve` stays CPU — the integration gate stays valid (GS identical on CPU in
  both paths).
- **DO NOT touch / port** `cpp/src/wave_solver.cpp` — it is DEAD/orphaned (not in
  CMakeLists, superseded by `wave_substep`). Confirmed by the scout.
- No wave-physics change. The gate enforces bit-identity.

Mirror the established template (S3/S4): a `.cu` of grid-stride kernels + a plain-C++
`.h` + a backend flag (`wave_backend_is_cuda`/`set_wave_backend_cuda`) + a
`PhysicsEngine` dispatch + an isolated pybind gate binding + a subprocess pytest.
Read `cpp/src/cuda_smoke.{h,cu}` + `cuda_water.cu` as the working template.

---

## 1. The passes (mirror `wave_substep` — READ atmosphere_solver.cpp ~62-261)

Scout map (verify line numbers against the actual file). All gather-per-cell unless noted.

| # | pass | reads | writes | notes |
|---|------|-------|--------|-------|
| K1 | feed source (~116-124) | wave_source, (consts) | wave_p, wave_source | `feed = min(src·feed_rate·dt, src, max/step)`; `wave_p+=feed; wave_source-=feed`. Pure int. |
| K2 | Laplacian gather (~137-158) | wave_p (4-nbr), perm | `lap[]` scratch | per-face `w=quantize((double)min(perm[self],perm[n]))` (device bridge, like S4a neighbor_q_dev), `flux=mul_wide(w, wp[n]-wp[self])`, `lap=narrow(Σ4)`. OOB face=0. |
| K3 | velocity kick (~169-173) | lap[], wave_v | wave_v | `wave_v += narrow(mul_wide(c_sq_dt_q,lap) - mul_wide(damp_dt_q,wave_v))`. int64 widen. |
| K4 | pressure update (~175-178) | wave_v | wave_p | `wave_p += mul_q16(wave_v, dt_q)`. |
| K5 | absorption (~180-193) | wave_absorb, wave_p, wave_v | wave_p, wave_v | `a=mul_q16(quantize(wave_absorb),absorb_str_dt_q); k=(a<FP_ONE)?FP_ONE-a:0; wave_v=scale_mag(wave_v,k); wave_p=scale_mag(wave_p,k)`. |
| K6 | wave BCs (~195-201) | masks | wave_p, wave_v | `=0` where obstacle/wall/vacuum. |
| K7 | **mean_wp reduction** (~204-220) | wave_p, masks | a device int64 accumulator | **see §2** — the determinism crux. |
| K8 | anomaly transfer (~223-260) | wave_p, mean_wp (scalar) | atmosphere | `anom=wave_p-mean_wp; d=round_nearest(anom·xfer_q); atmosphere += d` (one-sided; wave NOT drained). Sign-symmetric round. |

Host scalar precompute (in the entry fn, double, verbatim from the CPU top): the
quantized step constants (`feed_rate·dt`, `c_sq·dt`, `damp·dt`, `dt`, `absorb_str·dt`,
`xfer`, the thresholds). Pass as scalar kernel args. Device scratch: `lap[]`, the int64
accumulator. Separate kernel launches = barriers between dependent passes. Every thread
writes its own cell fully (no uninitialised scratch).

---

## 2. The `mean_wp` global reduction — the determinism crux (the arc's FIRST GPU reduction)

CPU (atmosphere_solver.cpp ~204-220):
```
interior_mask[i] = !obstacles[i] && !is_wall[i] && !is_vacuum[i]
sum = mean_sum(wave_p, interior_mask, n)     // int64, order-free
count = #interior
mean_wp = mean_round(sum, count)             // round-half-away-from-zero, NO pre-shift
```
GPU strategy (plan §1.3/§1.7, proven on spike-0a):
- **K7 reduce:** a device `int64` accumulator `d_sum` (cudaMalloc + memset 0). Each thread:
  `if (interior[i]) atomicAdd((unsigned long long*)d_sum, (unsigned long long)(int64_t)wave_p[i]);`
  (atomicAdd on 64-bit; integer `+` is **associative + commutative → order-free → the
  final sum is bit-identical regardless of thread/scheduler order**, unlike float. No
  overflow: |sum| ≤ ~2^51, int64 has headroom — verify against the field bound.)
  Compute `count` on the host (a host pass over the masks, or a second int atomicAdd) —
  it's deterministic either way.
- **Host:** read back `d_sum`; compute `mean_wp = mean_round(sum, count)` ON THE HOST
  (the exact CPU `mean_round`: `(sum>=0)?(sum+count/2)/count:(sum-count/2)/count`, q16).
- **K8 transfer:** receives `mean_wp` as a scalar kernel arg.
- The interior membership predicate is **bool topology only** (no float) → identical.

This int64-atomicAdd reduction is the NEW technique. The gate must exercise it hard
(varied interior masks, ± wave_p, near the accumulator's magnitude) and confirm the
GPU sum == the CPU sum exactly.

---

## 3. New shared device helpers (add to `cpp/src/cuda_fixedpoint_device.cuh`)

These are reused by S7 (sponge/GS), so put them in the shared header (NOT file-local):
- `__device__ int32_t scale_mag_dev(int32_t x, int32_t k)` — magnitude-first signed
  shrink, verbatim of `fixedpoint::scale_mag` (fixed_point.h:318): `x==0?0 : sign(x)*((|x|*k)>>FP_SHIFT)`.
- `__device__ int32_t round_nearest_q_dev(int64_t prod)` — the sign-symmetric
  round-to-nearest used by the anomaly transfer (and S7's GS increment): mirror the CPU
  `(prod>=0)?((prod+HALF_Q)>>FP_SHIFT):-(((-prod)+HALF_Q)>>FP_SHIFT)` (find the exact CPU
  form in atmosphere_solver.cpp's transfer + fixed_point.h; replicate it bit-for-bit).
Adding NEW functions doesn't change existing helpers — but rebuild + confirm the S3/S4
gates still pass (they don't call the new helpers, so they must be unaffected).

`reciprocal_q16_dev` / `shr_round0_dev` are NOT needed for S5 (they're for the S7
GS/wind). `mul128_shr_signed` is NOT needed (no 128-bit products in wave_substep).

---

## 4. Files
- `cpp/src/cuda_wave.{h,cu}` (NEW) — the kernels + the reduction + host entry + backend flag.
- `cpp/src/cuda_fixedpoint_device.cuh` — add `scale_mag_dev`, `round_nearest_q_dev`.
- `cpp/CMakeLists.txt` — add `src/cuda_wave.cu` to the BREACH_CUDA list.
- `cpp/src/physics_engine.cpp` — wrap the `atmos.wave_substep(...)` call in the n_wave
  loop (~210-221) in `#ifdef BREACH_HAS_CUDA / if (wave_backend_is_cuda()) {
  breach_cuda::wave_substep_gpu(...) } else #endif { ... }`. Include `cuda_wave.h` guarded.
- `cpp/src/bindings.cpp` — `set/get_wave_backend` + a `cuda_wave_substep(...)` isolated
  gate binding (mirror cuda_smoke_step; pass the scalar dials explicitly since it's a free
  function — read the live `wave_substep` binding for the arg list).
- `tests/cuda_s5_check.py` (NEW, mirror cuda_s4a_check.py):
  - **PART 1 isolated:** synthetic wave_p/wave_v/wave_source/atmosphere + wave_source(>thresh)
    + wave_absorb + perm + masks; run `wave_substep` n times CPU vs GPU on identical copies;
    `np.array_equal` tol 0 on wave_p AND wave_v AND wave_source AND atmosphere. MUST hit:
    the source feed, the lap with varying perm, the absorb (scale_mag), the BCs, AND the
    **mean_wp reduction with varied interior masks + ± wave_p** (assert the GPU sum/mean
    matches), the anomaly transfer; degenerate grids; many seeds.
  - **PART 2 integration:** a shockwave scenario (seed wave_source) through both
    `PhysicsEngine` wave backends via `set_wave_backend()`; full per-tick trajectory of
    wave_p/wave_v/wave_source/atmosphere bit-identical over 30 ticks; default-scenario CPU
    digest still `60bd331f…`. Print `S5_RESULT: PASS`/`FAIL`.
- `tests/test_cuda_s5_wave.py` (NEW) — pytest wrapper.

---

## 5. Build + gate
Same as S4 (`cpp/build_cuda.bat`; interpreter `C:/Users/steen/anaconda3/python.exe`;
gate → `S5_RESULT: PASS`; full suite). Bit-identity tol 0 IS the oracle → auto-merge on
green. **Top risk = the mean_wp int64 reduction** (order-freedom + the host round) — prove
the GPU sum equals the CPU sum exactly across the isolated configs. Secondary: the absorb
`scale_mag` (magnitude shrink) and the anomaly-transfer sign-symmetric round.
