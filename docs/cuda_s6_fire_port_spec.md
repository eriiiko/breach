# CUDA-S6 — FireSimulation::step GPU port spec

**Status:** in progress (branch `cuda-s6-fire`).
**Goal:** a faithful, **bit-identical** GPU port of `FireSimulation::step`
(`cpp/src/fire_simulation.cpp` ~44-267) — ignition/spread (logistic) → plume pressure
→ smoke emission → wall burn-through → clamp. The synced fields `fire`, `atmosphere`,
`smoke` (gas plane), `wall_hp` (int32 Q16.16) must come out **byte-for-byte identical**
CPU vs GPU (tol 0); the returned **destroyed-walls list must match as a SET** (order is
irrelevant — the caller processes each cell independently; only field state is synced).

**CONFIRMED by scout:** FireSimulation has **NO RNG** — pure deterministic integer
Q16.16. So this is a pure gather/scatter port, no PRNG needed. No fire-physics change.

Mirror the S1/S3/S4/S5 template + the shared `cuda_fixedpoint_device.cuh`.

---

## 1. The passes (mirror `FireSimulation::step` — READ fire_simulation.cpp ~44-267)

| # | pass | shape | reads | writes | notes |
|---|------|-------|-------|--------|-------|
| P1 | early-exit (~61-66) | HOST max | fire | — | `if max(fire) < thresh_q: return {} ` and DO NOT modify fields. Do this on the **HOST** (`*max_element(fire)`) before launching — bit-identical to the CPU host max; skip all kernels + return the H2D'd fields unchanged (= empty destroyed). |
| P2 | logistic spread (~111-192) | gather | fire, temperature, wind_x/y, atmosphere(4-nbr mean), wall_hp, masks, flammable | fire | the **pinned LEFT-FOLD `mul_q16` tree** (per-thread sequential — naturally deterministic), `sqrt_q16_dev` for wind magnitude, `mean_round` 4-nbr atmosphere, **snap-extinguish** `if I_next<I_min: 0`. |
| P3 | plume pressure (~194-215) | own-tile | fire (P2-updated), atmosphere | atmosphere (in-place) | `gain = round(...)`; `if (gain>0) atmosphere[i]+=gain`. Own-index → no race. |
| P4 | smoke emission (~217-237) | **SCATTER → 4 nbrs** | fire (P2-updated) | smoke (4 neighbours) | `smoke[nbr] += round_nearest(emit(fire[src]))` for non-wall neighbours. **Use integer `atomicAdd`** (order-free → bit-identical to the CPU's sequential adds). **VERIFY** the deposit depends only on `fire[src]` (NOT on `smoke[nbr]`) — if it did, it would be order-dependent; the scout says it's source-only, confirm in the code. |
| P5 | wall burn-through (~239-256) | own-tile + **list scatter** | fire (P2-updated), wall_hp, flammable, masks | wall_hp, fire(=0 on destroyed), the destroyed list | `wall_hp[i] -= round(...)`; `if (wall_hp<=0 && flammable && is_wall) { append (i/w,i%w) to destroyed; fire[i]=0; }`. Collect destroyed via a **device atomicAdd counter** + an index array (order arbitrary — gate is SET equality). |
| P6 | final clamp (~258-264) | own-tile | fire, smoke | fire, smoke | clamp both ∈ [0, FP_ONE]. |

**Pass order (separate kernel launches = barriers):** P2 → P3 → P4 → P5 → P6. P3/P4/P5
all read the P2-updated `fire` (frozen between launches); P5 zeroes `fire` on destroyed
cells AFTER P3/P4 have read it; P6 clamps last. Host scalar precompute: all config
constants quantized once in double (verbatim from fire_simulation.cpp's load-time
block, lines ~68-104 — many use load-time `make_recip`/`recip_mul`, NOT per-cell
`reciprocal_q16`). Pass them as scalar kernel args.

---

## 2. The new device helper + the scatter/collection patterns

- **`sqrt_q16_dev`** — add to `cuda_fixedpoint_device.cuh`: a verbatim device port of
  `fixedpoint::sqrt_q16` (fixed_point.h, the floor-isqrt Newton loop). Pure integer,
  fixed iteration count — replicate it EXACTLY (same iterations, same rounding). Used by
  P2 for the wind magnitude `sqrt_q16(wx²+wy²)`.
- **P4 atomicAdd scatter:** each source thread computes its 4 rounded emissions and
  `atomicAdd`s each into the neighbour's `smoke` (cast through the int32→… per the field
  type). Integer atomicAdd is associative/commutative → order-free → the per-neighbour
  sum is bit-identical to the CPU's sequential row-major adds. (This is the S5 reduction
  insight applied to a scatter; S2 raycaster already used a saturating-int atomic, so the
  pattern is precedented — but here it's a plain non-saturating add.)
- **P5 destroyed-list collection:** a device `int` counter (atomicAdd to get a slot) +
  a device array of packed indices; copy `counter` + the array back to the host; build the
  `std::vector<std::pair<int,int>>` from it (any order). The gate checks set equality.
  Size the device array at `n` (worst case every cell destroyed).

`scale_mag_dev`/`round_nearest_q_dev`/`reciprocal_q16_dev` already exist; `narrow_round_signed`,
`mean_round`, `mul_q16`, `recip_mul` are FP_HD (callable on device) — reuse, don't reimplement.

---

## 3. Files
- `cpp/src/cuda_fixedpoint_device.cuh` — add `sqrt_q16_dev` (verbatim of fixedpoint::sqrt_q16).
- `cpp/src/cuda_fire.{h,cu}` (NEW): the P2-P6 kernels + the host entry `fire_step(...)` that
  does P1 (host max early-exit), H2D, the kernel chain, the destroyed-list D2H, returns
  `std::vector<std::pair<int,int>>`. Backend flag `fire_backend_is_cuda`/`set_fire_backend_cuda`.
- `cpp/CMakeLists.txt` — add `src/cuda_fire.cu` to the BREACH_CUDA list.
- `cpp/src/physics_engine.cpp` (~104-111): wrap the `this->fire.step(...)` call in
  `#ifdef BREACH_HAS_CUDA / if (breach_cuda::fire_backend_is_cuda()) { destroyed =
  breach_cuda::fire_step(...) } else #endif { destroyed = this->fire.step(...) }`. The GPU
  entry RETURNS the destroyed vector (unlike S1-S5 which only mutate fields). Include
  `cuda_fire.h` guarded.
- `cpp/src/bindings.cpp` — `set/get_fire_backend` + a `cuda_fire_step(...)` isolated gate
  binding (mirror the live `FireSimulation.step` binding that returns a py::list of pairs;
  pass the scalar dials explicitly).
- `tests/cuda_s6_check.py` (NEW, mirror cuda_s5_check.py):
  - **PART 1 isolated:** synthetic fire(incl. 0 → P1 early-exit, and large → P2 feedback),
    temperature(± ), wall_hp(some tiny → P5 burn-through), atmosphere(edge cases), wind(±,0
    → sqrt path), masks, flammable; CPU `FireSimulation().step` vs `cuda_fire_step`;
    `np.array_equal` tol 0 on fire AND atmosphere AND smoke AND wall_hp, AND
    `set(destroyed_cpu) == set(destroyed_gpu)`. MUST hit: P1 early-exit, P2 logistic +
    sqrt + snap-extinguish, P4 smoke scatter **with overlapping neighbour deposits**
    (two fire cells emitting into a shared neighbour — proves atomicAdd order-freedom),
    P5 burn-through (destroyed set), degenerate grids, many seeds.
  - **PART 2 integration:** a seeded-fire scenario (fire burning + oxygen) through both
    `PhysicsEngine` fire backends via `set_fire_backend()`; full per-tick trajectory of
    fire/atmosphere/smoke/wall_hp bit-identical over 30 ticks; default-scenario CPU digest
    still `60bd331f…`. Print `S6_RESULT: PASS`/`FAIL`.
- `tests/test_cuda_s6_fire.py` (NEW) — pytest wrapper.

---

## 4. Build + gate
Same as S5. Bit-identity tol 0 (+ destroyed SET equality) IS the oracle → auto-merge on
green. **Top risks:** (1) the P4 atomicAdd scatter (prove order-freedom with overlapping
deposits); (2) `sqrt_q16_dev` matching the host floor-isqrt exactly; (3) the destroyed-list
device collection (no drops/dupes — gate set-equality + length catches it); (4) pass-order
(P5 zeroes fire AFTER P3/P4 read it). The P2 mul-fold is per-thread sequential → naturally
deterministic (low risk). DO NOT change fire physics; reuse FP_HD helpers.
