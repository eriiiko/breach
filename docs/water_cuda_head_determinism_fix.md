# Water CUDA head-term determinism fix (EOS-P3 collateral)

**Date:** 2026-07-11
**Branch:** `eos-p6-close` (commit 1, ahead of the P6-close cleanup)
**Scope:** `cuda_water.cu`, `cuda_water.h`, `bindings.cpp` (`cuda_water_step`),
`physics_engine.cpp` (`step_water` GPU dispatch), `tests/cuda_s3_check.py` +
`tests/test_cuda_s3_water.py`.

## One-line summary

The EOS refactor's P3 patch rewrote the **CPU** water pressure-head term from a
float bridge to a pure-integer `mul_q16(k_p, P)` and retired `wave_p`, but the
**CUDA** water kernel was never updated — it kept computing the head via the old
float bridge on a float `atmosphere`/`wave_p`. That is a real cross-GPU
determinism break whenever `k_p != 0` (pressure-head ON). This patch reconciles
the CUDA kernel + its binding + the live GPU dispatch + the S3 gate with the P3
integer-P contract, restoring bit-identical CPU-vs-GPU water.

## What P3 changed (CPU), and what the CUDA kernel still did

**Before P3** — both paths computed the head as a FLOAT BRIDGE:

```
surface += quantize( k_p * (atmosphere_float + wave_p_float) )
```

`atmosphere` and `wave_p` were float head fields; the product was formed in
float and requantized into the Q16.16 surface.

**After P3 (design §6 "water head")** — the CPU (`water_solver.cpp:117-132`)
became a pure-integer head on the derived integer pressure `atmosphere` (== P,
Q16.16), and the phantom `wave_p` arg was dropped (P already carries the acoustic
transient + the bulk dome as one merged field):

```cpp
const q16 kp_q = quantize((double)k_p);
...
if (head_on) {
    const q16 atm_v = atmosphere ? atmosphere[i] : 0;   // integer P
    s += mul_q16(kp_q, atm_v);                          // PURE INTEGER
}
```

The **CUDA** kernel (`cuda_water.cu`, `water_surface`) was left on the pre-P3
float bridge:

```cpp
if (head_on) {
    const float atm_v = atm_f ? atm_f[i] : 0.0f;
    const float wp_v  = wave_f ? wave_f[i] : 0.0f;
    const float head_f = kp_f * (atm_v + wp_v);
    s += quantize((double)head_f);                      // FLOAT BRIDGE
}
```

Float-dequantize → multiply → requantize is a different rounding path than the
integer `mul_q16`, so the surface potential — and everything downstream (the
velocity kick, donor-cell fluxes, depth divergence) — diverges in the LSBs on any
head-on cell. `atmosphere`/`wave_p` even had different *types* across the two
binding signatures (`cuda_water_step` took float + `wave_p`; `WaterSolver.step`
took int32 P + no `wave_p`), so no gate could feed both the same input.

## Why it was latent (and why the P6 close surfaced it)

- The live default is `WATER_K_P = 0.0` (`physics_runner.py:62`, "W4 turns it
  on"). With `k_p == 0` the head term is skipped on **both** paths, so the live
  water backend stayed bit-identical and nothing failed in normal play.
- During P3 the live GPU water dispatch was deliberately stubbed rather than
  fixed: `physics_engine.cpp` guarded `water_backend_is_cuda()` with
  `assert(false && "EOS P3: cuda water head bridge retired; port pending P6")`.
  (In a Release/NDEBUG build that `assert` compiles out, which would have
  *silently skipped* the water substep had anyone flipped the backend on — a
  second reason the stub had to become a real port.)
- The S3 water gate — which explicitly exercises `k_p = 0.5` head-on configs —
  had been SKIPPED since P6.0 (the whole-suite CUDA pin `cuda_available()` returns
  False while any kernel is pinned). Closing P6 (unpinning the last kernel,
  `combustion`) re-activates that gate, which is when the divergence would have
  turned red. It was caught during the P6-close cleanup instead.

Water is outside the EOS refactor proper, so this port had no key in
`EOS_P6_PENDING_KERNELS`; it was tracked only by the `assert`/"port pending P6"
marker. Erik's decision was to FIX it now (restore GPU bit-identity coverage)
rather than retire the gate.

## The fix

1. **`cuda_water.cu` (`water_surface` kernel):** head branch rewritten to the
   integer form `s += mul_q16(kp_q, atm_p[i])`, matching the CPU exactly.
   `kp_q = quantize((double)k_p)` is precomputed on the host in `water_step` (the
   same host cast the CPU does) and passed to the kernel as a `q16`. The float
   `atm_f`/`wave_f` params and the `kp_f` float are gone; `atmosphere` is now a
   nullable `const int32_t*` (Q16.16 P). Device buffers/copies for `wave_p` are
   removed; `d_atm` is sized as int32.
2. **`cuda_water.h`:** `water_step` signature drops `wave_p` and changes
   `atmosphere` from `const float*` to `const int32_t*`; header comments updated.
3. **`bindings.cpp` (`cuda_water_step`):** `atmosphere` is cast to
   `py::array_t<int32_t>`, the `wave_p` param + plumbing are removed, matching the
   live `WaterSolver.step` binding.
4. **`physics_engine.cpp` (`step_water`):** the `assert(false)` GPU stub is
   replaced with a real `breach_cuda::water_step(...)` call, forwarding the same
   int32 `atm_bridge` (== P) the CPU reads plus the solver's scalar dials.
5. **`tests/cuda_s3_check.py`:** the synthetic `atmosphere` is now an int32
   Q16.16 P field (both-sign, modest scale so the head term fires non-trivially);
   `wave_p` is removed from `_make_inputs` and from both the `cpu.step` and
   `cuda_water_step` call sites. Docstrings updated from "FLOAT BRIDGE" to the
   integer head term.

`mul_q16` is the shared FP_HD fixed-point multiply; the device implementation
(`cuda_fixedpoint_device.cuh`) is bit-identical to the host `mul_q16`, so with the
same integer P and the same host-computed `kp_q` the head — and therefore the
whole solver — is byte-for-byte identical CPU vs GPU.

## How it is now proven

The S3 gate (`tests/cuda_s3_check.py`, run via `tests/test_cuda_s3_water.py`)
proves bit-identity CPU-vs-GPU on `water_depth` + `flow_vx` + `flow_vy` (tol 0):

- **PART 1 (isolated):** 45 synthetic configs incl. **both** `k_p = 0` (head off)
  and `k_p = 0.5` (head ON — the configs that were diverging), across degenerate
  1xN/Nx1 grids, the outflow limiter, tilt poly, and the dry/solid/eps clamps.
  The head-on configs passing bit-identical **is** the proof the head fix works.
- **PART 2 (integration):** the full-engine `set_water_backend(True/False)`
  trajectory over 30 ticks is byte-identical, and the CUDA build's CPU path still
  reproduces the committed default-scenario golden
  `98d3dd7eaf3d574d6e562513cd95f3b5ac077b7c69b1d0b024db931261735473`.

This runs green as part of the P6-close full suite on the Ada (Lenovo) box with
the CUDA build present.
