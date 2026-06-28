# CUDA-S4b — SmokeDynamics::sink_hop GPU port spec

**Status:** in progress (branch `cuda-s4b-sinkhop`).
**Goal:** a faithful, **bit-identical** GPU port of `SmokeDynamics::sink_hop`
(`cpp/src/smoke_dynamics.cpp:309-355`) — the decoupled breach sink-pull (one 1-cell
BFS-gradient hop). Completes the GPU smoke path (S4a did `step`; this does `sink_hop`).
The synced `gas` field (int32 Q16.16) must come out **byte-identical** CPU vs GPU
(tol 0) on every plane after the hop.

This is a **small, reuse-heavy** patch. `sink_hop` is structurally identical to S4a's
advection: snapshot `src` → per non-(obstacle|wall|vacuum) cell back-trace into `src`
→ clamp. It **REUSES the already-verified** `backtrace_sample_q_dev` and `smoke_clamp`
from `cuda_smoke.cu`. The ONLY new logic is the displacement:

```
double sink_disp = min((double)sink_strength, 1.0);   // host scalar
bx_q = quantize(sink_disp * (double)sink_x[i]);        // per-cell FLOAT BRIDGE
by_q = quantize(sink_disp * (double)sink_y[i]);
smoke[i] = backtrace_sample_q_dev(src, x, y, bx_q, by_q, ...);
```
vs S4a's advection `bx_q = -mul_q16(wind_x[i], dt_adv_q)`. Everything downstream of
`(bx_q,by_q)` is the SAME verified machinery.

## Determinism
- `sink_disp = min(sink_strength, 1.0)` is computed ONCE on the **host** in double
  (it's a scalar), passed to the kernel.
- The per-cell `bx_q = quantize(sink_disp * (double)sink_x[i])` runs on the device
  in **double** (`sink_x` float → double, × double `sink_disp`, `quantize`). Rely on
  `--fmad=false` (already on all CUDA TUs) — same float-bridge pattern as S4a's
  permeability/wind² bridges. Character-identical to the CPU expression (lines 335-338).
- With no breach, `sink_x/sink_y` are all-zero → `bx_q=by_q=0` → `backtrace_sample_q_dev`
  is the identity (sealed rooms untouched) — the gate must cover this.

## Files
- `cpp/src/cuda_smoke.cu`: add a `__global__ smoke_sink_hop` kernel (per-cell: skip
  obstacle/wall/vacuum → keep `src` value; else compute the sink `(bx_q,by_q)` →
  `backtrace_sample_q_dev`). Add a host entry `void smoke_sink_hop(int32_t* smoke,
  const float* sink_x, const float* sink_y, const bool* obstacles, const bool* is_wall,
  const bool* is_vacuum, const float* perm, int h, int w, float sink_strength)` that
  does ONE hop: H2D + D2D snapshot `src=smoke` + the sink kernel + the `smoke_clamp`
  kernel + D2H. (One entry call = one hop, exactly mirroring the CPU; the engine loops
  K=`vent_hops` times. Per-call H2D/D2H; residency/K-fusion is S8.)
- `cpp/src/cuda_smoke.h`: declare `smoke_sink_hop(...)`.
- `cpp/src/physics_engine.cpp`: in the K-hop sink loop (~337-352), wrap the
  `this->smoke.sink_hop(...)` call in `#ifdef BREACH_HAS_CUDA / if
  (breach_cuda::smoke_backend_is_cuda()) { breach_cuda::smoke_sink_hop(...
  this->smoke.sink_strength) } else #endif { this->smoke.sink_hop(...) }`. The **same**
  `smoke_backend_is_cuda()` flag now gates BOTH `step` (S4a) and `sink_hop` (S4b), so
  `set_smoke_backend(True)` routes the whole smoke path to the GPU. Update the binding
  comment that said "sink_hop ALWAYS stays on the CPU" (no longer true).
- `cpp/src/bindings.cpp`: a `cuda_smoke_sink_hop(...)` isolated gate binding (mirror
  `cuda_smoke_step`; args = smoke + sink_x/sink_y floats + masks + perm + `sink_strength`).
- `tests/cuda_s4b_check.py` (mirror cuda_s4a_check.py):
  - **PART 1 isolated:** synthetic smoke + sink_x/sink_y fields (incl. a breach-gradient
    pattern producing nonzero ±displacements AND an all-zero/sealed case = identity),
    masks, several grid sizes incl. degenerate, many seeds. `SmokeDynamics().sink_hop`
    vs `cuda_smoke_sink_hop` on identical copies; `np.array_equal` tol 0.
  - **PART 2 integration:** a scenario **WITH a breach** (so sink_hop actually pulls)
    through both `PhysicsEngine` smoke backends via `set_smoke_backend()` — now routing
    BOTH step and sink to the GPU — full per-tick `gas` trajectory bit-identical over 30
    ticks; default-scenario CPU digest still `60bd331f…`. Print `S4B_RESULT: PASS`/`FAIL`.
- `tests/test_cuda_s4b_smoke_sink.py` (mirror test_cuda_s4a_smoke.py).

## Build + gate
Same as S4a (`cpp/build_cuda.bat`; interpreter `C:/Users/steen/anaconda3/python.exe`;
gate → `S4B_RESULT: PASS`; full suite). Bit-identity tol 0 IS the oracle → auto-merge
on green. NOT in scope: any smoke-physics change; reimplementing `backtrace_sample_q_dev`
(reuse it).
