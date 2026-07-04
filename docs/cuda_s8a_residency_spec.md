# CUDA-S8a — GPU residency (the payoff stage)

**Status:** in progress (branch `cuda-s8a-residency`).
**Goal:** keep the synced physics fields **GPU-resident across the whole tick** — allocate
device buffers once, run the entire tick's kernels on them in place (no per-call malloc /
H2D / D2H), and copy the whole synced set down to a CPU mirror **once per tick** (the locked
Q4 decision) so combat/render/recorder read the mirror exactly as today. This deletes the
per-call transfer tax the benchmark found (~0.6–0.95 ms/call). CUDA graphs are S8b (next),
render CUDA-GL interop + recorder kernels are S8c (deferred).

**Proven foundation (the de-risk spike — `cpp/src/cuda_spike.*`, `tests/_spike_s8.py`):**
- A kernel given a CuPy array's `int(arr.data.ptr)` (passed as `uintptr_t` through pybind)
  mutates that CuPy-owned device memory IN PLACE — CuPy + the breach `.pyd` share the one
  CUDA primary context. Contiguous arrays + C-contiguous planes (`gas[k]`) work with a plain
  indexed kernel; non-contiguous views need an explicit element-stride (we keep fields
  contiguous). `.data.ptr` is stable unless the array is reassigned/reallocated.
**Reuse that `reinterpret_cast<int32_t*>(dev_ptr)` pattern.**

**Bit-identity is preserved end to end:** the resident path must produce byte-for-byte the
same synced fields as the existing per-call GPU path (which is already == the CPU). No
physics change.

---

## 1. The architecture

- **Persistent device buffers = the GameMap synced fields, as CuPy arrays** (Python-owned,
  allocated once, C-contiguous, written ONLY in place). The full synced set (the ~14 int32
  Q16.16 fields + the masks the kernels read): atmosphere, wave_p, wave_v, wave_source,
  wind_x, wind_y, gas (N,h,w), fire, temperature, heat, water_depth, flow_vx, flow_vy,
  wall_hp, + the read-masks obstacles/solid/is_vacuum/flammable + dyn_permeability/face_shift/
  heat_inv_shift/floor_height (confirm the exact set the kernels touch by reading the entries).
- **Persistent scratch = C++-owned** device buffers (the per-solver temporaries: water's
  surface/fx/fy/dq_e/dq_s/scale; smoke's lap/src; diffuse's rhs/dinv/vac_a/vac_b; wave's lap +
  the reduction accumulator; the float bridges atm_f/wave_p_f; etc.). Allocate once (lazily on
  first resident call, keyed by h*w), reuse every tick. NEVER malloc per call in the resident path.
- **A resident orchestrator** — a new `PhysicsEngine` entry (e.g. `step_resident(...)`) that
  takes the field device pointers (from Python via `.data.ptr`) + the scalars + the substep
  counts, and runs the WHOLE tick's kernel sequence on the persistent buffers, mirroring the
  EXACT order + arithmetic of the current `step_water` → `run_substeps` → `step_tail` (+ the
  raycaster `cast_fire_heat`). The substep counts (n_water/n_wave/n_smoke) stay HOST-computed
  (integer ceil_div — control flow) and are passed in.
- **The per-tick seam:** Python (1) ensures host-produced inputs are on device (the stamped
  masks obstacles/dyn_permeability, the FieldEdit deltas — H2D once per tick, OR keep them
  resident + a small scatter), (2) calls `step_resident` with the field device pointers, (3)
  downloads ALL synced fields to numpy mirrors (`gmap.to_host()` — one batched D2H) so the
  Python post-physics (combat/recorder) + render read the mirror unchanged.

---

## 2. Build steps (the agent does these; INVENTORY first, report scope early)

**STEP A — inventory the tick (READ `cpp/src/physics_engine.cpp`: `step_water`, `run_substeps`,
`step_tail`; + `src/simulation/physics_runner.py` for `cast_fire_heat` + the call order).**
List EVERY inter-solver per-cell array op that currently runs on the host pointers between the
solver kernel calls — the float bridges (dequantize atmosphere/wave_p → atm_f/wave_p_f for the
water head), the W5 flash-boil water→steam sink, the W3 displacement/seal, any field swaps/copies,
the gas-plane striding, etc. **Each of these becomes a device kernel** in the resident path (so
nothing round-trips mid-tick). **Report this inventory + the count EARLY** (it sizes the patch) —
if it's much larger than ~a-handful, stop and report before building the rest.

**STEP B — extract "launch-only cores".** For each of the 7 solver `.cu` files, factor out a
function that takes persistent device pointers (fields + scratch) + scalars and ONLY launches
the kernels (no malloc/H2D/D2H/free). Keep the existing per-call `*_step` entries intact (the
gates + the per-call dispatch still use them — the live fallback). The launch-only core is the
shared kernel-launch body.

**STEP C — write the glue kernels** (from STEP A's inventory) — the bridges/flash-boil/etc. as
device kernels on the persistent buffers, bit-identical to the current host C++ ops.

**STEP D — the resident orchestrator** `step_resident(...)`: owns/lazily-allocates the persistent
scratch; runs cast_fire_heat-launch → step_water-launch (n_water substeps + flash-boil + seal) →
run_substeps-launch (n_wave wave + diffuse + n_smoke smoke + sink) → step_tail-launch (ripple +
fire + temperature), all on the passed-in field device pointers + the persistent scratch. Same
order, same substep counts, same arithmetic as the host path.

**STEP E — GameMap CuPy + the seam** (`src/simulation/gamemap.py`): a residency mode where the
synced fields are CuPy device arrays (contiguous), allocated once; `gmap.to_host()` (one batched
D2H of all synced fields to pinned numpy mirrors that the existing Python reads use); a
`device_ptrs()` accessor (the `.data.ptr`s for `step_resident`); and a `__setattr__` GUARD that
makes REASSIGNING a resident field (`gmap.fire = x`) a hard error (in-place `gmap.fire[:] = x`
only — stale-pointer protection). The CPU path + the existing numpy fields must still work when
residency is OFF.

**STEP F — dispatch + flag.** A new backend mode (e.g. `set_residency(True)` / a
`PHYSICS_BACKEND=resident`) that routes the tick through `step_resident` + `to_host()` instead of
the per-call path. Default OFF (the game + suite unchanged). Wire it in `physics_runner.step` /
`Simulation.step`.

---

## 3. The gate (bit-identity + the payoff)
- `tests/cuda_s8a_check.py` (mirror the prior gates) — **PART 1**: a full-engine A/B over **30
  ticks** on the canonical seeded scenario, residency ON vs the CPU path, asserting the per-tick
  trajectory of ALL synced fields is bit-identical (`diff_trajectories` tol 0), AND the
  residency path reproduces the golden `60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1`.
  (Residency == per-call GPU == CPU, transitively.) **PART 2 (the payoff)**: a benchmark — time
  N ticks residency vs CPU at a couple of grid sizes; print the speedup and confirm the per-call
  malloc/transfer tax is gone (residency should beat the per-call GPU clearly, and beat CPU at
  the grids where the benchmark predicted). Print `S8A_RESULT: PASS`/`FAIL`.
- `tests/test_cuda_s8a_residency.py` — pytest wrapper.
- Full suite green: `… -m pytest tests/ --ignore=tests/test_main_smoke.py --ignore=tests/test_renderer_smoke.py`.

---

## 4. Discipline + honesty
- **Bit-identical, tol 0.** No physics change. Keep the per-call path + the CPU path as the live
  defaults (residency behind the flag).
- **In-place only** for resident fields (the `__setattr__` guard enforces it).
- If the STEP-A glue inventory reveals this is much bigger than ~a-handful of kernels, or you
  cannot land a GREEN full-tick gate, **STOP, report the inventory + exactly what's done vs
  blocking, and DO NOT commit a broken state** (leave the branch building, the per-call path
  intact). A partial-but-green slice + an honest report beats a sprawling half-state.
- Build: `cpp/build_cuda.bat`; interpreter `C:/Users/steen/anaconda3/python.exe` (has cupy 14.1.1);
  CPU build `cpp/build/Release`. Reuse `tests/cuda_harness.py`.

---

## 5. THE BUILD TARGET — the residency SLICE (Erik chose this; inventory complete)

The glue inventory (done) found the tick has TWO ops that are NOT simple field→field kernels:
**`cast_fire_heat`** (a Python per-burning-tile raycaster loop that writes the synced `heat`,
*first* in the tick) and **`step_ripple`** (render-only float, has NO `.cu` kernel; writes the
synced `ripple`/`ripple_v`). Porting those to device is multi-day. So the SLICE keeps them on
their **existing host paths** and makes only the **tax-heavy field→field block** resident.

**The resident block (in tick order):** `_step_water` (water substeps + W5 flash-boil + W3
displacement/seal) → `run_substeps` (wave×n_wave → diffuse → smoke×n_smoke → sink×K) → the
`step_tail` tail of **fire → temperature**. These are ALL existing bit-identical GPU kernels
(S1–S7) — the slice just runs them on persistent CuPy buffers instead of per-call malloc/transfer.

**The host-path outliers stay where they are**, run on the host mirror at their correct tick
positions: `cast_fire_heat` (before the block, writes `heat`) and `step_ripple` (the ripple part
of step_tail, before fire/temp; render-only, never feeds back into transport).

**Per-tick boundary sequencing (a HANDFUL of transfers, vs per-call dozens):**
1. `cast_fire_heat` runs on the host mirror (as today) → updates `heat` (+ light/glow) on host.
2. **H2D once:** upload the synced set the resident block reads (incl. the freshly-written `heat`)
   into the persistent CuPy buffers. (Also upload the per-tick host inputs: stamped masks
   `obstacles`/`dyn_permeability`, applied FieldEdit deltas — they're produced host-side pre-physics.)
3. **Resident block** runs entirely on the CuPy buffers: water(+W5+W3) → wave×n → diffuse →
   smoke×n → sink → fire → temperature, using the launch-only cores + the glue kernels (bridges,
   flash-boil, displacement/seal) — NO per-call malloc/transfer inside.
4. **D2H `water_depth`** (only) so the host `step_ripple` can read it; run `step_ripple` on the host
   mirror (it writes `ripple`/`ripple_v`, render-only — fire/temp already ran and don't read ripple,
   so ordering is preserved: ripple is computed from the post-transport `water_depth`, exactly as the
   CPU does it at the top of step_tail before fire/temp... CONFIRM the exact ripple position in
   step_tail vs fire/temp and replicate it — if ripple must precede fire/temp, split step_tail so
   ripple(host) runs between the substeps and the resident fire/temp).
5. **D2H once:** download the full synced set → the numpy mirrors, so the Python post-physics
   (combat/recorder) + render read the mirror unchanged (the Q4 baseline).

**Determinism note:** because `cast_fire_heat` + `step_ripple` run the SAME host CPU code in both
the residency path and the CPU-reference path, and the resident block runs the SAME kernels as the
per-call GPU path (already == CPU), the full trajectory is bit-identical — *provided the boundary
H2D/D2H sequencing places every host op on the right field values and the resident block sees the
host ops' outputs*. That sequencing is the crux; get it exactly right and verify against the golden.

**Launch-core extraction caveat:** the 7 solver kernels are in anonymous namespaces inside their
`.cu` files. Extract each solver's launch sequence into a NON-anonymous `*_launch_resident(...)`
(device-pointer args + persistent scratch, no malloc/transfer), declared in a shared internal
header the resident orchestrator includes. Keep the per-call `*_step` entries calling the same
cores (or unchanged) — the per-call path + the gates must still pass.

**Scratch:** the resident orchestrator owns the per-solver scratch device buffers (water's
surface/fx/fy/dq_e/dq_s/scale; smoke's lap/src; diffuse's rhs/dinv/vac_a/vac_b; wave's lap +
reduction accumulator; the float bridges atm_f/wave_p_f) — allocate once (lazily, keyed by h*w),
reuse every tick. Never per-call malloc in the resident path.

**Gate (§3) unchanged:** the 30-tick A/B (residency ON vs CPU) must be bit-identical on ALL synced
fields incl. `heat`/`ripple`/`ripple_v` (the host-path outliers must come out identical too) +
reproduce the golden, and the benchmark must show the per-call tax gone on the resident block.
