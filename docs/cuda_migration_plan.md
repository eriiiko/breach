# Breach CUDA Migration Plan

**Status: PLAN — synthesized from an autonomous 6-stream research run (2026-06-27), then walked with Erik decision-by-decision. The 7 open questions + the substep-cliff item are now LOCKED (see "Decisions (locked 2026-06-27)" below and §8). NOT yet implemented.**

> **Decisions (locked 2026-06-27).** Erik walked the plan's open questions one at a time; all are resolved. The headline structural consequences:
> 1. **The substep-cliff integerization is a pre-CUDA BEDROCK PATCH, not "CUDA-S0."** It completes the integer foundation (water already proves the `ceil_div` pattern) and lands *before* the CUDA arc begins. The CUDA arc's first step is now the first **kernel** (temperature). The cliffs are double-but-correctly-rounded-deterministic *today* (not a live desync) but un-integerized — the patch finishes them. See §1.4 + §7.
> 2. **Toolchain (Q1):** the **already-installed CUDA 12.4** + `-allow-unsupported-compiler` (VS2022 17.14 is newer than 12.4 officially lists; the flag is harmless for pure-integer kernels). Erik raised his DRIVER to **610.62** (lifts the CUDA ceiling) but we **stay on the 12.4 toolkit** — no toolkit download. Same setup on the Lenovo (Ada) later. See §4.
> 3. **Clean cpp/build (Q2):** reset the poisoned VS18-2026 cmake cache onto VS2022 via a **targeted** reset (delete `CMakeCache.txt` + reconfigure — **not** a recursive `rm -rf`, respecting the deny-list); prune stale `C:/tmp` worktrees. See §4.0.
> 4. **CuPy (Q3):** chosen, **installed, and working** (`cupy-cuda12x` 14.1.1 on `numpy` 2.4.6; full Breach suite **369 green**; numpy-2 is fine for Breach). Coexists with the future PyTorch (ML); CuPy↔PyTorch share GPU memory via `__cuda_array_interface__`. See §2.4.
> 5. **Combat/gameplay field read (Q4):** **copy ALL fields GPU→CPU each tick** as the baseline (full gameplay access; ~50 µs at the 50×120 grid — effectively free; correctness/access first). The integer fields ARE deterministic gameplay state, so gameplay reads broadly. Then **OPTIMIZE per-system** — migrate a mature system's logic into a GPU kernel so it reads on-device, shrinking the per-tick copy over time. NOT a subset-download (transfer **latency** dominates, not bytes, so batching the whole small field beats fragmented per-element gather; gather is only for huge sparse grids — Breach isn't). See §2.2.
> 6. **mean_wp (Q5):** port the deterministic **stopgap** (integer global reduction, deterministic via integer atomics) to GPU FIRST — a faithful bit-identical CPU→GPU translation (changing the physics mid-port would break the bit-identity gate). The **edge-flux retirement is a COMMITTED, TRACKED post-port milestone** with its own name (§7.7), not a footnote. See §1.3 + §7.
> 7. **CUDA graphs (Q6):** one dedicated optimization pass AFTER all solvers are ported + bit-identical (not per-kernel). Pure speed, correctness-first. See §7.
> 8. **Scope (Q7) — IMPORTANT CORRECTION:** the **RAYCASTER IS IN SCOPE.** It is fixed-point and its heat rays **inflict damage** (heat deposit → the integer `heat` field → `combat.apply_environmental_damage` → unit HP), so it is **deterministic gameplay physics, not render-cosmetic** — and the **most parallelizable kernel** (embarrassingly-parallel independent rays). It becomes an **early CUDA kernel** (after temperature de-risks the toolchain, because it carries a scatter-atomic wrinkle). Its gameplay-affecting integer output (`heat`, a scatter → integer `atomicAdd`) is gated on bit-identity; the purely-visual float outputs (`light_rgb`/`light_dir`/`smoke_glow`) stay float-OK (render-local). Only the **combat-HP kernels remain separate** (the Q2-fenced HP math → the future Q2-lift). The arc ends when the physics solvers **+ the raycaster** are on GPU + graph-optimized. See §0.2, §0.3, §3.8, §7.

This is the direct sequel to the shipped fixed-point arc (`docs/fixed_point_migration_plan.md` +
`docs/s1_water_fixed_point_plan.md` / `docs/s2_fixed_point_plan.md` / `docs/s3_fixed_point_plan.md`,
**annotated** tag **`bedrock`** on commit `bbe359c`). That arc converted the **entire** sim-state path — water,
atmosphere, wave, wind, smoke, multi-gas, fire, temperature — to integer **Q16.16** and proved it
**bit-identical on the CPU** across compilers/toolchains (337+ tests green, `field_ab_harness.py`
`tol=0.0`). It also de-risked the two load-bearing GPU kernel shapes on real Ampere hardware
(`spike0/`, the `bedrock`-era spike): an integer reduction and an integer Red-Black Gauss-Seidel,
both **GPU-int == CPU-int bit-for-bit**, while the float versions jittered.

This plan is **work planning over a locked design, not new design.** The GPU-residency architecture is
already canon in `docs/architecture/engine/02_state_and_ownership.md` ("the `GameMap` interface stays;
the solver methods become CUDA kernels; the hot fields go GPU-resident"). What remains is to *sequence*
the port, *gate* it kernel-by-kernel against the CPU integer oracle, and *prove the cross-GPU
reproduction* on Erik's own Ampere (RTX 3070, sm_86) and the incoming Ada Lenovo (sm_89).

> **Locked decisions inherited from the fixed-point arc (do not relitigate):** full fixed-point Q16.16
> for synced state; cross-GPU bit-identity is the contract; NVIDIA-only is acceptable; render/cosmetic
> buffers stay float; the SYNCED-vs-LOCAL boundary is the real determinism boundary; `temperature_solver`
> is the production template; integer reductions are order-free (so `atomicAdd`/`atomicMax` on int are
> safe); no float atomics, ever, on the synced path.

> **The one-line thesis.** The fixed-point arc converted the *arithmetic* to something a GPU reproduces
> bit-for-bit. This arc moves that arithmetic onto the GPU and *proves the reproduction on real
> Ampere + Ada hardware*, **one kernel at a time**, with the CPU integer solver kept as a live fallback
> so the game never stops running.

---

## 0. Orientation + scope

### 0.1 What this plan is, and is not

| It IS | It is NOT |
|---|---|
| A sequenced work-order to move the shipped integer C++ solvers onto CUDA kernels behind the existing `PhysicsEngine` seam | A redesign of the physics — **no new math.** The CPU integer result is the oracle; a kernel that changes a single bit is a bug, not a feature |
| A determinism contract: device field bytes == CPU reference field bytes, bit-for-bit, on Turing (7.5) / Ampere (8.6) / Ada (8.9) | A float performance port. The arithmetic is integer; that is what makes cross-GPU bit-identity **free** rather than a 20–30% Reproducible-FP tax |
| A residency + transfer model (fields live on GPU; tiny deltas up; one snapshot per frame down) | A "copy everything every substep" naïve port — that PCIe-bottlenecks the whole sim (§2) |
| A CUDA pedagogy path — one new GPU concept per step, because Erik is learning CUDA | A throughput-first plan. Determinism-provability is the gating axis; perf (CUDA graphs, shared-mem tiling) comes *after* correctness |

This supersedes the GPU *sequencing* of the old `docs/cuda_integration_plan.md` (2026-03-25), which
predates fixed-point and led with "diffusion + raycaster first" purely for throughput. Post-`bedrock`
the gating axis is **determinism-provability**, so the order inverts for the *diffuse* solve: the
reduction-coupled RB-GS-with-a-barrier — the hardest thing to prove bit-identical — moves to **last**
(leaf stencils first — §7). The **raycaster still lands early** (Q7), but for the new reason, not the old
one: it is now in-scope deterministic gameplay physics (heat → damage) and the *most parallelizable*
kernel, so it follows the first stencil (temperature) once the plumbing is trusted — determinism +
parallel-cleanliness, not raw throughput. The old plan's GPU *mechanics* (memory hierarchy, kernel
patterns) remain valid background reading.

### 0.2 What ports to the GPU

The eight shipped solver translation units, all integer Q16.16, behind the `PhysicsEngine` seam
(`run_substeps` / `step_water` / `step_tail` / `stamp_units`):

- `water_solver` (donor-cell shallow water + render-only `step_ripple`)
- `atmosphere_solver` — `wave_substep` (explicit wave) **and** `diffuse_solve` (implicit Red-Black GS)
- `smoke_dynamics` (semi-Lagrangian advection + diffusion, the 5-plane gas group)
- `fire_simulation` (per-cell logistic + scatters)
- `temperature_solver` (conduction stencil)
- `physics_engine` orchestration (host-side; issues the substep loops + the reductions)
- `stamp_units` (per-tick dynamic-field rebuild from the unit footprints)
- `raycaster` — **IN SCOPE, an EARLY kernel** (Q7 locked 2026-06-27). It is fixed-point and its heat rays
  **inflict unit damage** (the integer `heat` deposit → `combat.apply_environmental_damage` → unit HP),
  so it is **deterministic gameplay physics, not render-cosmetic** — and it is the **most parallelizable
  kernel** (embarrassingly-parallel independent rays). The **heat deposit is a SCATTER** (multiple rays →
  one cell → integer `atomicAdd`, deterministic); its gameplay-gated integer output (`heat`) is gated on
  bit-identity. The purely-visual float outputs (`light_rgb`/`light_dir`/`smoke_glow`) stay float-OK
  (render-local). See §3.8, §7.6.

**NOT in scope — dead code:** `cpp/src/wave_solver.cpp` (`WaveSolver`) is an **orphaned float
pressure-wave TU** — it is **not** in the `pybind11_add_module` source list (`cpp/CMakeLists.txt:23-32`
omits it; it appears there only in a `/fp:strict` comment as "render-only"), and `WaveSolver` is **never
instantiated**. It was superseded by `atmosphere_solver::wave_substep` (the real, integer wave path this
plan ports under §3.2). **DEAD — delete, do not port.** Do not mistake it for the wave solver.

### 0.3 What STAYS on the CPU (the SYNCED-vs-LOCAL boundary, ported intact)

Two categories, exactly as the fixed-point arc fenced them:

1. **The Q2-fenced Python combat/HP path.** `combat.py` runs on the host and reads current-tick
   `heat`/`temperature`/`atmosphere`/`fire`/`solid`/`is_vacuum`, then writes back `fire`. **It reads
   integer fields but does its HP/damage math in float64** (`apply_environmental_damage`,
   `combat.py:194-218`: `dmg = env_rate * (…) * dt_tick`; `u.current_hp -= dmg` — Python scalar float).
   This is **Erik's deliberately-ratified Q2 fence**, not an open blocker: combat HP/damage stays
   Python-float for now (a Python scalar float `+−×÷` is same-machine reproducible — no FMA, no
   transcendental jitter), and **S3c already ships the unit-state determinism digest**
   (`tests/test_s3c_unit_state_digest.py`) that hashes per-unit HP + the hit/kill event stream, so the
   fenced float HP is **watched end-to-end, fire→heat→kill**. (`apply_blast_damage` already quantizes its
   damage to `int`, `combat.py:130`.) The path is *Python*, *serial*, and *actor-shaped*; it stays on the
   CPU and reads the integer fields off device. **The baseline transfer is the full-field copy GPU→CPU
   each tick (Q4 locked, §2.2)** — the integer fields ARE deterministic gameplay state, so gameplay reads
   broadly and the whole small field set comes down each tick (~50 µs at the 50×120 grid — effectively
   free; transfer **latency** dominates, not bytes). Migrating combat's *math* into device kernels is the
   **future Q2-lift**, deliberately deferred — and is **out of this arc's CUDA field-port scope.** When Q2
   is lifted it will additionally need integerizing `dmg`/`current_hp` *and* the digest-tightening the S3c
   review flagged (the 1e-9 HP quantum + the incomplete `SYNCED_UNIT_FIELDS` set). It is not a residency
   prerequisite (§3.7).
2. **The render/cosmetic float.** `light_rgb`, `light_dir`, `smoke_glow`, `ripple`, `light_map`, and the
   `dyn_*` / `permeability` / `light_atten` solver-coefficient floats. These are **not** Q16.16, **not**
   lockstep state, and deliberately stay float (the fixed-point arc left them float on purpose). They
   are render-only (produced and consumed on-device via CUDA-GL interop, §2.1) or per-tick derived
   coefficients (0-ULP copy/min/max, safe by construction).

Also CPU-resident: the editable `material` source of truth (`destroy_wall` runs on host), the entity /
projectile / RNG / tick-phase state ("GPU owns the fields, CPU owns the actors"), the FieldEdit queue
(branchy, RNG-ordered, deterministic stable-sort — must stay host-serial), and the substep-count
integers (computed host-side from a deterministic reduction, uploaded as integers — §1.4).

### 0.4 The hardware reality

| Machine | GPU | Arch | cc | Driver | Role |
|---|---|---|---|---|---|
| Work desktop | RTX 3070 | Ampere GA104 | **8.6** (`sm_86`) | 560.94 | primary dev; Ampere gate **now** |
| Lenovo laptop (incoming) | Ada | Ada Lovelace | **8.9** (`sm_89`) | (current line) | migration target; Ada gate **in a few weeks** |
| (opportunistic) | Turing | Turing | **7.5** (`sm_75`) | — | third cross-arch data point (64-bit emulation) |

The fat binary targets **both** `sm_86` and `sm_89` (+ `sm_75` opportunistically) so one `.pyd` runs on
every machine and the cross-arch digest gate (§1.5) has real hardware to run on.

---

## 1. The DETERMINISM CONTRACT (CPU↔GPU bit-identity)

The contract is the same two falsifiable properties the fixed-point arc shipped, plus the GPU's new
failure modes closed by five rules. **Device field bytes == CPU reference field bytes, bit-for-bit, on
sm_75 / sm_86 / sm_89.**

- **P1 — cross-order / cross-arch bit-identity.** The same int32/int16 bytes regardless of reduction
  order, warp size, block count, scheduler, or GPU architecture.
- **P2 — conservation to the LSB.** Conserved fields (atmosphere, water — smoke/gas are
  deterministic-but-non-conservative by the locked SL decision) do not drift over a long settle.

The GPU port may only weaken these via a **new** failure mode that does not exist on the CPU. The five
rules below close every such mode. This maps to NVIDIA CCCL's strictest tier — *"GPU-to-GPU
determinism: results will always be bitwise identical for identical invocations, no matter the GPU"*
([CCCL #5550]). Float reductions reach that tier only via the special Reproducible-FP path (~20–30%
cost); **integer reductions reach it for free** because the non-determinism CCCL names comes from "the
non-associativity of floating-point arithmetic and the lack of order of execution guarantees from
parallel execution" — a defect integer add does not have. The spike measured this on Erik's own 3070:
integer Q16.16 → int64 `atomicAdd` was **bit-identical across all 20 repeats**
(`raw=-1514247643326`); float `atomicAdd` **varied every repeat**.

### 1.1 RULE A — The `__host__ __device__` shared-toolkit (bit-identity by construction)

Every arithmetic primitive on the synced path is **one function**, compiled for both targets by
annotating `cpp/src/fixed_point.h` `__host__ __device__`. The CPU reference and the GPU kernel call the
**identical token stream**; there is no separate "device version" to drift. This is the strongest
guarantee in the contract and it is nearly free, because two's-complement integer `+ − * >>` and
comparison are exact and associative with PTX-defined, architecture-independent semantics
(`mul.wide.s32` is an exact 64-bit product; `shr.s32` is a defined arithmetic shift). **Spike-0b is the
proof:** the integer RB-GS digest was `cpu_hash == gpu_hash == 0xAB27B2370160FFF4`, bit-for-bit, with
negative values present (signed `>>` exercised).

Mechanics:

1. **Annotate the toolkit.** Add `#define BX_HD __host__ __device__` (expands to nothing in a CPU-only
   TU) and prefix every inline: `mul_q16`, `mul_wide`, `narrow`, `narrow_round`, `narrow_round_signed`,
   `scale_mag`, `shr_round0`, `ceil_div`, `reciprocal_q16`, `mean_sum`/`mean_round`, `sqrt_q16`,
   `tan_poly`, `quantize`. They are already header-only `inline`/`constexpr` integer-only — written for
   exactly this port. Pin `-std=c++20` for **device too** (`nvcc -std=c++20`): the signed-shift
   portability argument depends on C++20. Add `static_assert(sizeof(q16)==4)` and a device-side
   `static_assert` on the standard.
2. **The one toolchain caveat — `recip_mul`'s 128-bit path is host-only; the device gets an explicit
   `__mul64hi` branch.** `recip_mul` is `#if __SIZEOF_INT128__ (clang/gcc `__int128`) / `_MSC_VER`
   (`_mul128`) / else`. **Do NOT rely on `__int128` on the device.** On Windows/MSVC device code 128-bit
   integers are **not available** (NVRTC docs; CCCL #1227) — the `__int128` branch is a *host* path
   (clang/gcc), not a device guarantee. **The PRIMARY `__CUDA_ARCH__` device path uses CUDA's
   `__mul64hi(a,b)`** (high 64 bits of a signed 64×64 product) + the low product, shifted by
   `RECIP_SHIFT` — this is the device high-multiply, present on every arch and not dependent on
   `__int128`. Keep `__int128` strictly as the **host** (clang/gcc) path. **Action: add the
   `__CUDA_ARCH__`→`__mul64hi` device branch and a host↔device cross-check unit test that `recip_mul(x,
   r)` returns identical bits for 10⁶ random `(x, divisor)` pairs, INCLUDING signed/negative `x` (the
   signed high-multiply is where `__mul64hi` and a 128-bit product can disagree if mis-handled).** This
   is the single function where "same source" is not literally true; everywhere else it is.
3. **`reciprocal_q16` is already the GPU-native divide.** Erik's integer-Newton `reciprocal_q16` (4
   fixed iterations, pure `mul_q16`) exists precisely so the GS `Dinv` and the SL renorm need **no
   hardware divide** — NVIDIA GPUs have no integer-divide instruction. Its fixed trip count makes it
   **branch-identical across all lanes** — warp-safe by construction. Same for `sqrt_q16` (32 fixed
   isqrt iterations). Both port verbatim.

**Residual risk (low):** a future caller adds a `<cmath>` call or a `double` constant inside a toolkit
function. *Mitigation:* extend the shipped no-float-in-sim-TU CI ratchet to scan the `__device__`-
compiled toolkit.

### 1.2 RULE B — No-fast-math / no-FMA-contraction (for every RESIDUAL float)

The sim field path has **zero float bridges left** (the arc's audited result). So Rule B governs the
**residual non-field floats**: the `dt` boundary cast, the load-time `double` bakes (`make_recip`,
config `quantize`, the tilt-slope `quantize((idx−c)*dx)` — flagged CPU-only-deterministic), the
`dyn_*` coefficient floats, and — defensively — anything a future patch reintroduces.

Rule B is mandatory because **float on the GPU is non-portable even at fixed reduction order.** Spike-0
method 4 proved this on one machine: `fmaf(w,x,acc)` gave `0xCEF16263`, separate `__fmul_rn` then
`__fadd_rn` gave `0xCEF16261` — **same math, two bits apart**, because the compiler chooses per-arch
whether to contract. NVIDIA's IEEE-754 whitepaper confirms FMA computes `rn(X×Y+Z)` with one rounding
versus `rn(rn(X×Y)+Z)` with two.

The rules for any residual float that touches synced state (or feeds an integer quantize):

1. **`nvcc --fmad=false`** on every TU compiling such float — the device analogue of the shipped MSVC
   `/fp:strict` + contraction-off. **Never `--use_fast_math`** (it implies `--fmad=true` plus
   reassociation, reciprocal substitution, flushed denormals). Pass the host-side `/fp:strict` through
   nvcc to cl with `-Xcompiler=/fp:strict`.
2. **`-prec-div=true -prec-sqrt=true -ftz=false`** so any residual `/` and `sqrt` stay IEEE
   correctly-rounded (these *are* portable cross-arch; it is FMA/reassociation/transcendentals around
   them that are not).
3. **The load-time-bake discipline (shipped — keep it for device).** Compute all `double` bakes once,
   **on the host, under `/fp:strict`**, then ship the *baked integers* to the device. The device never
   recomputes `make_recip` or a config quantize. **Hash the baked integer cache into the format-version
   tag** so two peers with divergent bakes fail loudly. The `dt` cast and integer substep counts are
   computed once on the host and uploaded as integers — the device never re-derives `n` from a device
   `ceilf`.
4. **No device transcendental on the synced path.** The only per-cell transcendental is `sqrt_q16`
   (integer isqrt); `tan_poly` is a scalar integer polynomial. Device `sqrtf` is 0-ULP but
   `sinf/cosf`≈2 ULP, `expf`≈2 ULP, `powf`≈4 ULP "may differ across architectures" — forbidden on the
   field path, allowed only in render/cosmetic.

**Residual risk:** the `dt`→integer cast and the tilt-slope bake run in host `double` and are
CPU-only-deterministic today. *Gate:* add both to the cross-machine digest's covered set; integerize if
they ever diverge (the tilt-slope re-derivation in pure integer is the preferred fix over `--fmad=false`
on the water TU, because `--fmad=false` over-kills legitimate FMA elsewhere).

### 1.3 RULE C — Deterministic parallel reductions (the big one)

**Claim, rigorously verified:** an **integer** `atomicAdd` to a wide accumulator is order-independent
and yields a single deterministic value regardless of thread/warp/block scheduling; a **float**
`atomicAdd` does not. *Proof:* `atomicAdd` serializes read-modify-writes into *some* scheduler-chosen
permutation π; the final value is `((…(0 ⊕ x_{π(1)}) ⊕ …) ⊕ x_{π(N)})`. Permutation-independence
requires `⊕` associative + commutative. Integer `+` (mod 2⁶⁴) is an associative abelian group operation
→ **every permutation gives identical bits.** Float `+` re-rounds each step → different π → different
bits. NVIDIA states this directly ([CCCL #5550]); the spike measured it on the 3070.

Two conditions must **both** hold for "deterministic via atomics for free":
- **(a) the accumulator never overflows** — `mean_wp` uses an int64 accumulator, worst-case
  `bits(sum) ≤ 51`, 12 bits headroom (fixed-point plan §4.2). int64, no saturation.
- **(b) the summand set is bit-identical to the CPU's** — the membership predicate is evaluated on
  **integer fields only** (no float-bridge `atmosphere < thresh`). The arc has no float bridges; masks
  are integer topology (`obstacles`/`is_wall`/`is_vacuum`).

**Per-reduction strategy:**

| Reduction | Operator | GPU strategy | Why deterministic |
|---|---|---|---|
| `mean_wp` (Σ `wave_p` over interior) | int64 `+` then `mean_round` | int64 `atomicAdd` to one accumulator (spike-0a pattern) **or** a fixed warp-shuffle→block→2nd-kernel tree; the divide is `mean_round` (round-half-away, **no pre-shift** — `sum` is already Q16.16) | integer add associative |
| `max_wind_sq` (feeds the `n_smoke` cliff) | `max` | any reduction shape, **integer** (the result feeds a substep cliff — §1.4) | `max` associative + idempotent |
| `max_fire` (early-exit), `boiling.any()`, `.any()` planes | `max` / OR | any shape, integer fields | order-free trivially |
| conservation sums (P2: Σatm, Σwater) | int64 `+` | int64 tree/atomic; harness/digest, not hot path | integer add associative |
| GS L∞ residual | `max |res|` | integer `max` reduction; diagnostic only | order-free |

**Decision rule:** `max`/`any`/`min` → cheapest reduction shape, no constraint beyond integer-field
membership. **Sum → must be integer into a non-overflowing wide accumulator; a float sum is forbidden on
the synced path.** Where a sum result crosses a control-flow threshold, it must additionally satisfy
§1.4.

**The deeper recommendation (locked, fixed-point plan Q8): retire `mean_wp` entirely — but AFTER the
port, not during it (Q5 locked 2026-06-27).** It is the only *sum*-reduction in the hot path, and it
forces a grid-wide barrier mid-tick (wave→reduce→broadcast→subtract→diffuse) — the real GPU cost is the
barrier, not the adds. The arc already ships the rounded int64 mean as the stopgap (`ac2cae8`); the
target replaces it with a **local edge-flux transfer** — order-free, barrier-free, no DC-bias,
conservative by construction. After that retirement the only hot-path reductions are `max`/`any`
(trivially deterministic) + the cliff inputs (§1.4). **This is the single biggest GPU-determinism +
performance win.** *Sequencing (Q5):* port the deterministic **stopgap** to GPU FIRST (a faithful
bit-identical CPU→GPU translation — changing the physics mid-port would break the bit-identity gate; the
stopgap is already shipped + golden, the integer global reduction is deterministic via integer atomics).
The edge-flux retirement is then a **COMMITTED, TRACKED post-port milestone with its own name — the
`mean_wp` edge-flux retirement (§7.7)** — landed once CUDA is up, where the CPU-shaped global-reduction
sync-barrier is replaced by the GPU-native local edge-flux. **Not a footnote; an explicit named
milestone.**

### 1.4 RULE C′ — The reduction-fed control-flow cliffs

`n_smoke = ceil(...)` over `max_wind_sq`, plus the atmosphere/wave `n` and the water `n`: a reduction
feeding an **integer substep count**. A 1-ULP slip flips `n` → two peers run a *different number of
iterations* → total desync a within-substep digest is **blind** to (fields are bit-identical *within* a
substep). **The shipped state is mixed, not uniformly integerized — state it accurately:**

- **WATER `n` is fully integer.** `step_water` (`physics_engine.cpp:350-352`) computes `n = max(1,
  ceil_div(sim_time_q, max_dt_q))` — `fixedpoint::ceil_div` (integer `(a+b−1)/b`) on Q16.16 inputs, no
  float64 cliff. This is the template the other two should reach.
- **The atmosphere/wave `n` and `n_smoke` are NOT `ceil_div` — they are double-but-deterministic.** The
  wave/diffuse `n` (`physics_engine.cpp:154-155`) is `max(1, (int)std::ceil((double)sim_time / dt))` with
  `dt = (double)atmos.max_dt()`. `n_smoke` (`:227-239`) computes `d_eff_max`/`dt_stable` in `double` then
  `ceil((double)sim_time / dt_stable)`. **Nuance worth keeping straight:** the `max_wind_sq` *reduction*
  that feeds `n_smoke` IS integer + order-free (`mul_wide` + integer `>` max, `:222-226`) — only the
  cliff *arithmetic* (the `ceil(sim_time/dt)`) runs in `double`. Because that `ceil` consumes
  bit-identical inputs (the integer max + config constants) through correctly-rounded IEEE `+−×÷`, it is
  **"double-but-deterministic"** — cross-platform bit-identical per the fixed-point arc's Lesson #1
  (correctly-rounded scalar `+−×÷` are bit-identical across platforms, no FMA/transcendental). So it is
  **NOT a proven live desync today** — but it is **un-integerized and UNVERIFIED cross-machine**, and a
  GPU kernel/host-launcher replacing it must reproduce it bit-identically.

**GPU rule (unchanged): the substep counts are computed host-side from integer inputs and uploaded as
integers; the device launches exactly `n` kernel iterations. No device `ceilf`, no device float CFL.**
The harness must drive ≥1 config where each cliff exceeds 1 and assert the count matches cross-arch.

**The BEDROCK PATCH (pre-CUDA, locked 2026-06-27) — integerize the atmosphere/wave `n` and `n_smoke`
cliffs like water.** This is **not** "CUDA-S0"; it is a **pre-CUDA bedrock-completion patch that lands
BEFORE the CUDA arc begins** (the CUDA arc's first step is then the first *kernel*, temperature — §7).
Convert atmosphere/smoke `max_dt()`→Q16.16, `d_eff_max`/`dt_stable`→integer, and compute `n` via
`fixedpoint::ceil_div` like water; regenerate the affected golden; Erik feel-check. It completes the
fixed-point arc (water already proves the `ceil_div` pattern), removes the last
double-but-only-correctly-rounded-deterministic cliffs from the synced control flow, and is the safest
ground for the GPU launcher. **Accurate framing (do not overstate):** the cliffs are double-*but*
correctly-rounded-deterministic *today* — not a live desync — just un-integerized; the patch finishes
the integer foundation. CPU-only change; lands first (§7, "Bedrock patch — first & likely last").

### 1.5 RULE D — Red-Black Gauss-Seidel on GPU (race-free by construction)

The atmosphere diffusion is **two kernel launches per sweep with a grid-wide barrier (separate launch =
implicit barrier) between colors**: launch-red updates only red reading only black; launch-black updates
only black reading only red. Within a single color there is **no intra-color read-after-write** — red
cells never read other red cells — so each cell reads a frozen input set and writes a disjoint address.
The result is **independent of the order red cells are scheduled** → identical on every architecture,
warp size, block count. It is a *fixed schedule you pin*, not a hazard you hope to avoid.

Specifics that preserve bit-identity:
- **Each cell written by exactly one thread** via a grid-stride loop over that color's cells → no write
  race, no atomics in the sweep.
- **Residual/flux form, not quotient form** (fixed-point plan §3.2): the increment is
  `atm[i] += mul(Σ_face flux − residual, Dinv[i])`. At the fixed point equal neighbours → increment
  truncates to exactly 0 → drift-free.
- **`Dinv[i] = reciprocal_q16(quantize(1 + μ·wsum))`** — the shipped integer-Newton reciprocal, no
  device divide. **On GPU, drop the CPU's incremental `dinv_key_` cache and recompute `Dinv`
  unconditionally per tick** — the key-compare `continue` is pure SIMT divergence, while
  `reciprocal_q16` is branch-identical and cheap. (The CPU "rebuild nothing" optimization *inverts* on
  GPU.)
- **Spike-0b is the on-hardware proof:** GPU integer RB-GS == CPU integer RB-GS bit-for-bit
  (`0xAB27B2370160FFF4`), 200 sweeps, 128², non-degenerate reciprocal (α=0.2), signed values present.

**Residual risk:** a future single-kernel grid-sync (`cooperative_groups::grid_group::sync()`) that
fuses the two colors must be digest-verified equal to the two-launch form on all three arches before
adoption (a block-level `__syncthreads()` does **not** synchronize across blocks → reintroduces a
cross-block RAW race). **Keep the two-launch form for the determinism proof.**

### 1.6 RULE E — Grid-stride-loop determinism

Every elementwise/stencil kernel uses a grid-stride loop where thread `t` owns output indices
`t, t+stride, t+2·stride, …` (`stride = blockDim.x*gridDim.x`). This makes the output→thread map a pure
function of the launch geometry and — critically — **each output element is written by exactly one
thread** ([NVIDIA grid-stride blog]). Consequences:
- **No write race → the written value is independent of scheduling**, exactly like the CPU loop.
- **Reads from the previous tick's buffer (ping-pong), not in-place**, for any neighbour stencil
  (advection, flux-apply, conduction, decay). RB-GS is the one in-place exception, safe *only* via the
  2-color schedule (§1.5).
- **Launch-geometry independence is a requirement** — the same kernel must produce the same bytes at 80
  blocks or 240. **The harness includes a geometry sweep:** run each kernel at ≥2 block counts and
  assert identical digests (catches a geometry-dependent indexing bug).
- **The conservative flux pair** (`field[i] += flux; field[n] -= flux`) survives the parallel write
  only if two threads don't race on `field[n]`. **Preferred: gather-per-cell** (each cell sums its 4
  own faces; one writer; no atomics) over edge-indexed `atomicAdd`. The shipped CPU gather-once/per-face
  shape ports verbatim; smoke/gas are non-conserved by the SL decision, so only atmosphere/water use the
  flux pair, and atmosphere diffusion *is* the RB-GS — the standalone flux pair is mainly water + the
  planned `mean_wp` edge-flux retirement, both designed as cell-owned gathers.

### 1.7 The reduction strategy in one paragraph (the answer to "how do we reduce deterministically")

Never a float atomic. For **sums**, use an integer (int64) accumulator that cannot overflow, reached by
either a single `atomicAdd` (order-free for integers) or a fixed hierarchical warp-shuffle→block→
2nd-kernel tree — both bit-identical to the CPU `mean_sum`. For **max/min/any**, use the cheapest
reduction shape (integer, idempotent → trivially order-free). The membership predicate is always on
integer fields. Any reduction result that then crosses a substep-count cliff is computed host-side via
`ceil_div` on the integer reduced scalar and uploaded as an integer. And, where possible, **retire the
one hot-path sum** (`mean_wp`) for a local edge-flux that needs no reduction or barrier at all.

---

## 2. MEMORY RESIDENCY + the host↔device transfer model

The canon contract (engine/02): **fields STAY on GPU through the whole physics step; Up = tiny deltas;
Down = one snapshot per frame; render may read stale, the sim never does.** Everything below realizes it.

### 2.0 The two clocks (the freshness invariant governing every boundary)

- **Per-tick, current-value, sim-authoritative reads.** `combat.py` reads `heat`/`temperature`/
  `atmosphere`/`fire`/`solid`/`is_vacuum` and writes `fire` *inside* `Simulation.step()`, after
  `physics_runner.step()` and before the next tick. Must see what the GPU just computed.
- **Per-frame, may-be-stale, render-only reads.** `game_renderer.upload_state()` reads the render
  fields once per *frame*, decoupled from the tick.

Different cadences → different transfers. The **headless training path has no frame** — it runs only the
per-tick clock and never produces the render colour buffers. **Baseline (Q4 locked, §2.2): the per-tick,
sim-authoritative read copies the WHOLE synced integer field set down each tick** (~50 µs at Breach's
grid — latency-bound, not byte-bound, so the whole-field batch beats a subset gather); the per-system
on-device migration that shrinks it is the later optimization.

### 2.1 THE RESIDENCY TABLE

"GPU-resident" = the device buffer is authoritative, allocated **once** at `GameMap.__init__`/`reset`,
written **in-place** (never reassigned — the in-place discipline is now a *device-pointer-stability*
requirement, and a **CUDA-graph-validity** requirement, §6). Cadence legend: `never` (lives/dies on
GPU) · `snapshot` (once-per-frame device→host, render only) · `tick-down` (the field is part of the
per-tick full-field device→host copy the sim reads — Q4 baseline; the whole synced set comes down each
tick, the `tick-down` rows just mark which fields gameplay actually consumes) · `delta-up` (host→device,
tiny, event-driven) · `static` (uploaded once at load / structural edit).

**Synced int32 Q16.16 fields — the bedrock state (GPU-resident, GPU-written):**

| Field | dtype | On-device writer | Synced WHEN |
|---|---|---|---|
| `atmosphere` | i32 | wave/diffuse/fire-plume/W3 | `tick-down` (combat O2/ignition) + `snapshot` (overlay) |
| `wave_p` | i32 | wave_substep, diffuse | `snapshot` only |
| `wave_v` | i32 | wave_substep | `never` (debug; recorder dumps on demand) |
| `wave_source` | i32 | consumed by wave_substep | `delta-up` (FieldEdit explosion deposit) |
| `wind_x`, `wind_y` | i32 | diffuse_solve | `never` (derived; dequantized on snapshot if shown) |
| `gas` (N,h,w) | i32 | smoke.step, sink_hop, W5 steam | `snapshot` (dequantize all N) + `delta-up` (FieldEdit) |
| `smoke` | view into `gas[BLACK_SMOKE]` | (via `gas`) | aliased sub-pointer (§4) |
| `fire` | i32 | fire.step logistic | `tick-down` (combat **writes** it back) + `snapshot` |
| `temperature` | i32 | temperature.step | `tick-down` (ignition + heat damage) + `snapshot` |
| `heat` | i32 | fire-heat raycast `atomicAdd`; cleared end-of-tick | `tick-down` (unit heat damage) |
| `wall_hp` | i32 | fire.step depletion | `tick-down` (burn-through → `destroy_wall`); `delta-up` on edit |
| `water_depth` | i32 | water.step, W5 boil | `snapshot` + `delta-up` (sources/FieldEdit); conserved |
| `flow_vx`, `flow_vy` | i32 | water.step | `never` (persistent solver state) |
| `floor_height` | i32 | none (read-only input) | `static` |

**Bool masks — small, structural, mostly static:** `solid`/`is_vacuum`/`flammable` are quasi-static
(`delta-up` on structural edit; combat keeps a host mirror of `solid`/`is_vacuum`, delta-updated, not
re-downloaded per tick). `obstacles` is GPU-rebuilt per tick by `stamp_units` → `never` synced (pure
solver input).

**Render-only float fields — GPU-resident only when a renderer is attached:** `light_rgb`,
`light_dir`/`light_dx,dy`, `smoke_glow` → **never downloaded** (CUDA-GL interop, §2.1a). `light_map`
(legacy scalar) → `snapshot` (units sample brightness) — small, retire with RGB migration. `ripple`,
`ripple_v` → `snapshot` (water pass).

**Per-tick dynamic solver-input caches — GPU-resident, GPU-rebuilt by `stamp_units`, `never` synced:**
`dyn_permeability`, `dyn_wave_absorb`, `dyn_light_atten`. **Float here is fine** — they are derived
solver inputs (structurally copy + min/max, 0-ULP, no reassociation) feeding *integer* synced fields;
the fixed-point arc deliberately left them float. This is the one place float survives in a per-tick
kernel and it is safe by the same argument the C++ already makes.

**Static caches — uploaded once, patched by delta on structural edit:** `material` (i8, host source of
truth + GPU mirror), `permeability`, `wave_absorb`, `light_atten`, `heat_atten`, `conductivity`,
`heat_inv_shift`, `face_shift` (h,w,4) — patched through the one `on_tile_changed` seam (+ 4 neighbours
for `face_shift`). **Re-plane `face_shift` (i*4) and `light_atten`/`dyn_light_atten` (h,w,3) into
separate coalesced planes on GPU** (§6) — the interleaved layout is stride-4/stride-3 and wastes
bandwidth. The `gas` (N,h,w) array is already plane-major — keep it.

**Pure host state — never touches the GPU:** `tilt_x`/`tilt_y` (kernel-arg scalars), `tile_size_m`,
`water_sources` (sparse list), `sink_x`/`sink_y` (lazy host BFS, `delta-up` only on topology change),
all entity/logic/RNG/tick/phase.

**Residency summary.** At Breach's ship grid (≈50×120) the full synced field set is tiny — the whole-set
per-tick download is **~50 µs (Q4 baseline)**; at 240×480 it is ~17 MB, at 1000×1000 ~150 MB. **VRAM is
not the constraint, and at Breach's size the download bytes are not either** (transfer latency dominates —
§2.2). The constraint that *does* matter is keeping per-tick host↔device traffic to the **one batched
per-tick full-field copy down + the deltas up** — every `never` row keeps that copy small, and every
`never` row means the whole `run_substeps`/`step_water`/`step_tail` chain executes on resident buffers
with no per-substep round-trip. (Per-system on-device migration later shrinks even the per-tick copy.)

### 2.2 TRANSFER BOUNDARIES (four, all tiny or event-driven)

**(A) The render read → CUDA-OpenGL interop (NOT a per-frame full download).** The fields are already on
the device; the raylib textures are also on the device. Download-to-reupload would be the dumbest
transfer in the program. **Register each render texture/PBO once** (`cudaGraphicsGLRegisterImage` /
`cudaGraphicsGLRegisterBuffer` — "registering a resource is costly… ideally only called once"),
**map per frame** (`cudaGraphicsMapResources` → `…GetMappedArray`/`…GetMappedPointer`), launch the
**pack kernels** (the dequantize `/65536`, the gas-plane→density, the smoke `^gamma` move here as device
kernels, killing the `game_renderer.py` CPU dequantize-scratch dance), **unmap**. This is **zero-copy**.
**Hard rule** (NVIDIA, explicit): *"accessing a resource through OpenGL while it is mapped produces
undefined results"* — the map→kernel→unmap window must fully bracket the pack; raylib must not touch the
texture until unmap returns. Wrap it in RAII. The render-only fields (`light_rgb`, `smoke_glow`,
`light_dir`) thus **never appear on the host** — produced on-device by the ray march, consumed on-device
by the pack kernels. Staleness is allowed here (one tick old is fine). One stubborn residual download:
`light_map` is host-sampled to light unit sprites — fold into a small device unit-brightness kernel, or
retire with the RGB migration.

**(B) The Python game-logic read → COPY ALL FIELDS device→host EACH TICK (Q4 locked 2026-06-27).**
`combat.py` (and any other host gameplay reader) runs on the host and reads current-tick values. It sits
behind the **Q2 fence** (combat HP/damage math is deliberately Python-float for now, §0.3 item 1, watched
by the S3c unit-state digest) — so the *transfer* below is in scope for this arc, but **migrating
combat's math onto the device is the later Q2-lift, not a residency prerequisite.**

**The baseline (locked): download the WHOLE synced integer field set once per tick.** The integer fields
ARE deterministic gameplay state (integer *because* they are in the lockstep loop), so gameplay reads
them broadly — not a hand-picked subset. At Breach's grid (≈50×120) the full field set is **~50 µs/tick —
basically free**; correctness/access first. One batched `cudaMemcpyAsync` from a pinned staging buffer
brings them all down; `fire` (the one field combat *writes*) goes back up as a `delta-up`.

**Why the whole field, not a gather of "the specific elements combat reads":** transfer **latency**
dominates, not bytes. Batching the entire small field into one copy beats fragmented per-element /
per-subset reads (each of which pays the launch+latency cost). A subset-gather is the wrong default — it
is only worth it for **huge sparse grids**, which Breach is not. Pick access/correctness now; the bytes
are not the constraint.

**Then OPTIMIZE per-system (the destination):** migrate a *mature* system's logic into a GPU kernel so it
reads on-device — `apply_environmental_damage` / `apply_temperature_ignition` move into device kernels,
combat reads device memory in a kernel, and that system's fields no longer need to come down. Each such
migration **shrinks the per-tick copy over time**. This is the future **Q2-lift** (it additionally
integerizes `dmg`/`current_hp` + tightens the S3c digest) — out of the field-port scope, sequenced after
the solvers + raycaster are resident. The full-field copy is the *correct, simple* baseline the port
ships on; per-system on-device reads are the *optimization* layered on later.

**(C) The recorder.** Disabled in headless training (no transfer). In debug, it needs 6 fields/tick
(`wave_p`,`wave_v`,`atmosphere`,`smoke`,`fire`,`obstacles`); two are already in (B). **Recommend a
device-side blowup check** (an integer `max|wave_p|` reduction kernel) that only triggers a full
device→host dump when it fires — steady-state cost is one reduction, not six downloads. Ring buffer
stays a flag-gated debug cost.

**(D) The field-edit queue → host→device DELTAS (the canonical write primitive).** `field_edit.py` is
the only sanctioned field-write path: sparse, host-authored, deterministic stable-sort drawing from
`sim.rng`. **Keep the queue, sort, RNG draw, and region iteration on the host** (branchy, serial,
RNG-ordered — the actor side; moving it to GPU would break the single-RNG-consumer guarantee). The flush
produces a small list of `(field, flat_index, quantized_delta, mode)` records (dozens to low-thousands,
not the whole grid). **Upload that sparse list** (one batched `cudaMemcpyAsync` from pinned memory) and
apply with a tiny **scatter kernel** per field — a direct port of `_combine_gas`/`_combine_heat`
(already integer Q16.16). **Ordering determinism is preserved** because the host sorts before upload;
within a field ADD is associative (order-free) and MAX is applied in the host-fixed order.
`water_sources` (continuous max-holds) are the same shape: a sparse host list → a tiny scatter-max
kernel.

### 2.3 PINNED MEMORY + the PCIe cost model

**The asymmetry that drives every decision:** device↔GPU bandwidth is hundreds of GB/s (~448 GB/s on
the 3070) while host↔device over **PCIe is ~16 GB/s (x16 Gen3)** — a **30–60× cliff**. NVIDIA:
*"minimize data transfer between host and device."* This is *why* the residency table pushes so much to
`never`.

- **Pinned (page-locked) host memory** (`cudaHostAlloc`/`cudaMallocHost`) for every recurring transfer
  (the per-tick full-field down, the delta-up, the recorder dump) — ">2× pageable" and avoids the
  driver's pageable→pinned bounce. **Caveat:** pinned memory is a scarce OS resource — pin **only** the
  staging buffers (the packed full-field staging region + a handful of (h,w) arrays), not redundant
  per-field mirrors.
- **Batch small transfers** — the per-tick down is **one** `cudaMemcpyAsync` of a packed staging region
  (the whole synced field set concatenated), **not** N per-field copies; the delta-up is **one** upload
  of the concatenated record list. The whole-field batch is *why* the Q4 baseline is latency-bound, not
  byte-bound: one copy pays one latency, a per-element/per-subset gather pays many.
- **The per-tick PCIe budget** (proving "no per-substep round-trip"): at Breach's grid the full-field
  Down (B) is **~50 µs/tick** (Q4 baseline); for reference, ~1.8 MB @ 240×480 / 16 MB @ 1000×1000; Up (D)
  ~KB. At 12 GB/s pinned, the 240×480 down is ~0.15 ms/tick; 1000×1000 ~1.3 ms/tick. A naïve port that
  downloaded all fields every *substep* (not tick) would be ~100–400× this and PCIe-bottleneck the sim —
  **that is the failure mode the residency table exists to prevent** (the Q4 baseline copies once per
  *tick*, not per substep). The per-system on-device migration shrinks even the per-tick copy.
- **Async overlap:** issue the per-tick down on a non-default stream so it overlaps GPU work where the
  dependency graph allows (the real overlap win is hiding the stale-tolerant render snapshot + recorder
  dump behind sim compute).
- **Do NOT use Unified/Managed memory (`cudaMallocManaged`) for the synced fields** — it makes residency
  implicit and page-migration-driven (the per-substep round-trips this stream exists to eliminate), and
  migration faults are nondeterministic in *timing* (not value, but they wreck the perf model). Explicit
  `cudaMalloc` + explicit boundaries keep residency *legible*, which is what both the perf model and the
  pedagogy need. (Managed memory is a fine future ergonomics experiment for the *static* caches, never
  the hot synced state.)

### 2.4 The `GameMap` interface → device-memory mapping

The canon promise: *"callers reach state through `gmap.<field>` and must never assume where it lives.
This makes the GPU migration a localized, mechanical change inside `GameMap` — no caller changes."*
Today every solver call does `get_2d(arr)` = raw pointer out of the numpy array, zero-copy, **re-fetched
every step** (cached nowhere — robust to reallocation). The CUDA port preserves this exact shape: the
solver methods take a **raw device pointer** instead of a raw host pointer. `int32_t*` is `int32_t*`
whether it points at host or device memory.

- Each `GameMap` field attribute becomes a **device-backed array object** owning a `cudaMalloc`'d buffer
  exposing `.device_ptr()` (the `get_2d` analogue, device-side — the C++ methods bind to it; the
  orchestration in `physics_engine.cpp` is unchanged in structure) and `.__array__()`/`.to_host()`
  (triggers the boundary download — snapshot for render-only, the per-tick full-field copy for
  combat/gameplay reads — returning a numpy view). **Existing host readers keep writing `gmap.fire`,
  `gmap.heat[cy,cx]`** — they transparently hit `.to_host()`. That is "no caller changes" in practice.
- **CuPy is the natural Python vehicle (Q3 locked — chosen, installed, working: `cupy-cuda12x` 14.1.1 on
  `numpy` 2.4.6, full Breach suite 369 green).** A `cupy.ndarray` *is* a device-backed array with
  `.data.ptr` (the pybind device pointer), `__cuda_array_interface__` (zero-copy agreement with the C++
  side — and the same interface by which CuPy will share GPU memory with the future PyTorch/ML stack),
  and `cupy.asnumpy()` (the `.to_host()` download). The gameplay fields are refreshed by the **per-tick
  full-field copy (B)** into a host mirror rather than per-element device fetch (per-element device
  indexing is catastrophically slow — the same latency argument that makes the whole-field batch the
  right baseline).
- **The `smoke = gas[BLACK_SMOKE]` aliasing survives** as a device sub-pointer `gas + BLACK_SMOKE*h*w`
  into the resident `gas` buffer — same as today's numpy slice-view; the "never reassign smoke"
  invariant becomes "never reassign the device sub-view." `gas` being C-contiguous means each plane is a
  contiguous device sub-buffer (the C++ already relies on `gas + gi*plane`).
- **The pybind boundary** gains a device-pointer overload: instead of `py::array_t<int32_t>` → `get_2d`,
  it accepts an integer device pointer (from `cupy_array.data.ptr`) or a `__cuda_array_interface__`
  capsule and passes it straight to the kernel-launching method. `get_2d`'s shape-extraction role
  (`h,w`) is kept; only the pointer's memory space changes. **This is the one mechanical change at the
  boundary the whole port turns on, and it is small.**
- **`material` stays host** (editable source of truth; `destroy_wall` runs there); its GPU mirror is a
  static cache patched by delta. So `GameMap` becomes *hybrid* — the §2.1 fields device-resident, the
  entity-adjacent fields host-resident — and the attribute interface hides which is which.

**Net:** the residency table + boundary spec collapse to a `GameMap`-internal change. No `combat.py`,
`game_renderer.py`, or `physics_runner.py` call-site rewrites are forced by *residency* (combat's
eventual move to device kernels in (B) is a separable *determinism* upgrade).

---

## 3. PER-SOLVER kernel mapping

The shipped Q16.16 state is the enabling precondition: **every reordering hazard below is solved for
free by integers.** Ship grid sizes are small (10²–10³ tiles/dim), so **occupancy and kernel-launch
latency dominate, not algorithmic FLOPs** — plan accordingly (§6 CUDA graphs).

### Master mapping table

| Solver (TU) | Kernel(s) | Race strategy | Reduction | Divergence | Hardest bit |
|---|---|---|---|---|---|
| **water_solver** `step` | K1 surface · K2 vkick · K3 gather face-flux · K4 flux→dq · K5 outflow-limiter · K6 scale-apply · K7 divergence · K8 clamp | gather-once per-face arrays → **no write race**; divergence reads faces, writes own cell | none | Low | the tilt `(idx−c)*dx` FMA hazard (§3.1) |
| **water_solver** `step_ripple` | splash · kick · drift (render-only float, double-buffered) | gather-then-apply | none | Low | none (not synced) |
| **atmosphere** `wave_substep` | feed · **lap (gather)** · vkick · pkick · absorb · bc · **mean_wp (reduction)** · transfer | lap→scratch; transfer one-sided (own cell) → no race | **mean_wp int sum** | Low–Med | the reduction (§1.3) |
| **atmosphere** `diffuse_solve` | **dinv** · **2×RB-GS/sweep** · residual · **vac-BFS ×2** · bc · wind | RB-GS two launches, race-free; BFS double-buffered | residual L∞ int `max` | Med | **the hardest solver** (§3.3) |
| **smoke_dynamics** `step`/`sink_hop` | diff (gather Lap) · **advect (SL back-trace gather)** · clamp | SL back-trace gathers from frozen snapshot → zero races | none | **HIGH** (DDA march + breach early-exit) | the divergent back-trace |
| **fire_simulation** `step` | **maxfire (reduction)** · logistic · plume(self) · **smoke-emit (1→4 scatter)** · walldamage + **destroyed-compaction** | logistic/plume own-cell; smoke-emit **integer `atomicAdd`** | max_fire int `max`; destroyed count | **HIGH** (sparse active set) | scatter + compaction (§3.5) |
| **temperature** `step` | heat→temp · **conduct (gather, double-buffered)** · cool | double-buffered gather → no race | none | **Low** | none — the clean first port (§3.6) |
| **physics_engine** orchestration | host loop launching the substep chains | n/a | **max_wind_sq int `max`** → `n_smoke` cliff | n/a | CUDA graphs (§6) |
| **physics_engine** `stamp_units` | reset-baseline · **stamp scatter** | min-perm/max-absorb/max-atten scatter → `atomicMin`/`atomicMax` (order-free) | none | Low | actor-shaped (keep host-built rows) |
| **raycaster** `cast_source_directional` (IN SCOPE) | **per-ray DDA march** · **heat deposit (multi-ray→cell SCATTER)** · render-float deposits (`light_rgb`/`dir`/`smoke_glow`) | per-ray parallel; **heat scatter → integer `atomicAdd`** (saturating, order-free); render floats are own-cell `+=` (race-y but render-exempt) | none (heat is a scatter, not a reduction) | **HIGH** (DDA march, data-dependent trip) | the scatter + the **float-then-quantize** heat path (gate the integer `heat`; RULE-B the float, or integerize) (§3.8) |

### 3.1 water_solver — already in the shape CUDA wants

The shipped code gathers face fluxes into per-face arrays `fx[]`/`fy[]`, narrows to per-face
`dq_e[]`/`dq_s[]`, then applies divergence — **the race-free design, no atomics.** Decompose into the
K1–K8 grid-stride chain. K7 is the conservation point: `dq_e[i]` is *the same value* removed from `i`
and added to `i+1` (read from a per-face array, not scattered) → conservation holds to the LSB on GPU
identically to CPU. **Two pre-port hazards:**
- **The tilt `(idx−c)*dx` double products** (water_solver.cpp:100–104) can fuse under device FMA → 1-ULP
  divergence. **Fix before GPU-residency: re-derive the tilt slope in pure integer** (preferred —
  surgical) or build the affected kernel `--fmad=false` (blunt; kills legitimate FMA elsewhere).
- **`flux_to_dq`/`recip_mul` use the `__mul64hi` device path, NOT `__int128`** (the `_mul128` branch is
  MSVC-host-only; `__int128` is not available in MSVC device code — RULE A.2). The device high-multiply
  is `__mul64hi` + the low product. **Add a device-vs-host golden** mirroring the existing
  `tests/_s1_flux_truncation_check.py` MSVC-vs-clang check, confirming the nvcc device `__mul64hi`
  truncation matches the host build bit-for-bit — **including the signed tilt difference** (`s_e − s_w`,
  `s_s − s_n`, `water_solver.cpp:159-160`), where the high-multiply sign-handling is exercised.

Shared-mem tiling of K1/K2's `surface[]` halo: **defer until profiling shows the stencil is
bandwidth-bound** (at ship grid sizes the field fits in L2).

### 3.2 atmosphere `wave_substep` — the first true reduction

Linear kernel chain; the Laplacian (`K-lap`) is the canonical FD shared-memory kernel (load tile + 1-cell
halo, `__syncthreads()`, read 4 neighbours from shared). The reduction (`mean_wp`) is the #1 determinism
hazard — but it is hard *only in float*; the integer sum is deterministic by construction (§1.3). The
transfer is one-sided forcing (own-cell write) → embarrassingly parallel. **In S5 port the `mean_wp`
STOPGAP faithfully (Q5)**; the local-edge-flux retirement that removes the reduction+barrier is the
post-arc §7.7 milestone (§1.3), not part of the S5 port.

### 3.3 atmosphere `diffuse_solve` — the HARDEST solver

The 2-color RB-GS = two kernel launches per sweep, race-free by construction (§1.5). **Do not** fuse
colors with `__syncthreads()` (block sync ≠ grid sync → boundary cells read stale neighbours). The
integer payoff is decisive: a *float* RB-GS on GPU would still be non-deterministic (each cell sums 4
float terms whose rounding depends on warp accumulation order); the integer `mul_wide`+`narrow` gather
is bit-identical regardless of scheduling. Convergence is identical (same `gs_iters`, same `Dinv`, same
round-to-nearest) — the GPU produces the same *partially-converged* field as the CPU, which is what
lockstep requires (we need *identical*, not *converged*). Sub-components: **K-dinv** (recompute
unconditionally — the CPU key-cache inverts to divergence on GPU); **K-residual** (integer `max`
reduction, diagnostic, skippable headless); **vac-dist BFS** (2 sequential ghost-passes, double-buffered
to break self-aliasing); **K-bc** (sponge relaxation); **K-wind** (`−grad(atm+wave_p)`, shared-mem
candidate). This is the roofline checkpoint — the int datapath is half-rate on Ampere (§9), so profile
occupancy/bandwidth here (Nsight).

### 3.4 smoke_dynamics — HIGHEST divergence, but zero races

The integer semi-Lagrangian back-trace is **embarrassingly parallel with zero races**: each thread
gathers from the frozen pre-pass snapshot (`src`, double-buffered) and writes its own cell. **No atomics,
no ordering hazard** — the ideal GPU pattern. But it is the worst warp-divergence offender: the DDA
wall-clip march has a *data-dependent trip count* + a `break` on a solid tile / breach. Mitigations:
(a) the march is *physically bounded* by the CFL substep (rarely >1–2 cells — the sink is capped at one
cell), so worst-case divergence is small in practice; (b) **convert the corner-validity tests to
predication** (`acc += valid * mul_wide(...)`; `wsum += valid * cw[k]`) instead of `continue` — NVIDIA:
*"use predication instead of branching"*, *"split phases into two loops"*; (c) adjacent cells have
similar wind → warps are largely coherent in march length. `sink_hop` is the same machinery, capped at
1 cell, run `K = vent_hops` times → host-driven launch loop. The diffusion Laplacian tiles in shared
memory; the advection gather reads scattered upwind cells → leave it on L2.

### 3.5 fire_simulation — sparse active set + a scatter + a compaction

`max_fire` is the easiest reduction (integer `max`, idempotent) — compute on device, branch the launch
decision on the host. The logistic is **high-divergence** (fire is <<1% of tiles): **stream-compact the
lit/flammable cells into a work-list and launch one thread per active cell** — the single biggest
fire-kernel win (eliminates divergence *and* slashes launch cost). `sqrt_q16` (fixed 32-iter) is
branch-identical → zero divergence. **`recip_mul` is per-cell HERE too** — the fire logistic calls it 4×
(`fire_simulation.cpp:124,128,155,206`), one of those on a **signed** difference (`:206`,
`FP_ONE − recip_mul(atmosphere,…)`). This is the same host-128-bit-vs-device path that risk #1 / §3.1
flag for water, so **the fire step (S6) must carry the identical device-vs-host `recip_mul` bit-identity
golden as the water step (S3) — explicitly including NEGATIVE/signed `(x, divisor)` inputs** (the fire
saturation term and the water tilt
`s_e − s_w` both feed signed values, where `__mul64hi` sign-handling differs from an unsigned high
multiply — the most likely place a naïve device port silently diverges). **The smoke-emit is a genuine
1→4 scatter race** (two adjacent fires write a shared neighbour) → **integer `atomicAdd`**
(order-independent, bit-deterministic — the integer foundation pays off again). The **destroyed-tile list is a stream compaction** (`atomicAdd` on a global
counter to claim a slot): the output *order* is non-deterministic but the *set* is deterministic — and
`destroy_wall` is order-insensitive set-application, so leave it unsorted (or sort the tiny list on
read-back). This is the one place a *count* reduction is needed.

### 3.6 temperature_solver — the clean first port

Three per-cell kernels: heat→temp (saturating add, own-cell), **conduct (the textbook conservative
gather stencil, already double-buffered** — reads the frozen pre-conduction field, writes a fresh one;
the `(tn−ti) >> face_shift` flux is shifted on the *difference* so equal neighbours give exactly 0 →
drift-free; **ping-pong the buffers, skip the copy**), and cool (4-neighbour vacuum-exposure test —
predicate the `break` as `exposed |= …`). **No reductions, no scatter, no conservation class.**
Essentially a drop-in port — which is exactly why it is the first kernel, S1 (§7).

### 3.7 physics_engine orchestration — host-driven, no kernels

`PhysicsEngine` becomes the **host orchestrator** issuing kernel launches over the GPU-resident fields.
The substep loops become **host-driven repeated kernel launches** on the same resident buffers (no
host↔device copy between substeps — the whole point). `max_wind_sq` (drives `n_smoke`) is an int64 `max`
reduction read back as one scalar; the cliff is `ceil_div` on the host (§1.4). `stamp_units` is
actor-shaped — **CPU builds the flattened stamp rows (as today), uploads the small `(ys,xs,perm,...)`
arrays, a stamp kernel scatters them with `atomicMin`/`atomicMax`** (or, since stamps are few, scatter
on CPU and upload the `dyn_*` fields — profile to decide; not on the hot field-solver path). Wrap the
per-tick launch sequence in a CUDA graph (§6).

### 3.8 raycaster — IN SCOPE, an EARLY kernel (Q7 locked 2026-06-27)

**This is gameplay physics, not render-cosmetic.** The integer `heat` deposit is **sim-affecting and
inflicts unit damage**: `combat.apply_environmental_damage` reads `gmap.heat` (`int(heat[ty,tx])` →
`phi = peak_raw / HEAT_SCALE` → felt-temp → `dmg` → `u.current_hp -= dmg`, `combat.py:155-247`). So the
raycaster's heat output is on the deterministic gameplay path and **must be bit-identity-gated.** It is
also the **most parallelizable kernel** — rays are independent (embarrassingly parallel), the easiest GPU
win — which is *why* it comes early (right after temperature de-risks the toolchain/plumbing, because it
carries one wrinkle the leaf stencils don't: the scatter).

**The kernel shape (read from `cpp/src/raycaster.cpp::march_ray_directional`):**
- **The ray-march is per-ray parallel** — one thread per ray (`cast_source_directional` already loops
  rays independently). DDA march, data-dependent trip count, `break` on aggregate cull / out-of-bounds /
  `max_range`. Bin rays by direction so a warp marches coherent rays (the §6 item 4 divergence mitigation).
- **The heat DEPOSIT is a SCATTER** (the one new concept vs temperature): multiple rays cross the same
  cell and all `+=` into `heat[idx]` (`heat_saturating_add(&heat[idx], …)`, `raycaster.cpp:218-221`).
  On GPU that is a **write race → use integer `atomicAdd`** (order-independent → deterministic; the
  saturating clamp becomes an `atomicMin`-against-`INT32_MAX` or a CAS-loop saturating add). The header
  itself anticipates this: *"Integer += is order-independent → deterministic … the property that lets
  `heat` become an atomicAdd on CUDA later"* (`raycaster.h:18-21`).
- **WRINKLE the leaf solvers don't have — the heat math is FLOAT-then-quantize, not pure integer.** Unlike
  the Q16.16 field solvers, the march computes `heat_dep = heat_emit * heat_survival * dist_atten` in
  **float**, then `heat_quantize()` rounds to Q16.16 and `heat_saturating_add` scatters the integer
  (`raycaster.cpp:219-220`). The *deposited integer* is the gameplay-gated output; the float that produces
  it is render-shaped per-machine math (the same same-machine-deterministic float class as the Q2-fenced
  combat HP). **Implication for the bit-identity gate:** the integer `heat` field is what P1 gates, but it
  is only bit-identical CPU↔GPU if the *quantize input* (`heat_dep`) reproduces — so the heat-channel
  float path on device must obey RULE B (`--fmad=false`, no `expf` contraction; note the gas-optics
  `std::exp` is light-only and does NOT touch heat, `raycaster.cpp:306-308`). Decide during S2
  whether to (a) hold the heat float path to `--fmad=false`/no-fast-math and gate the resulting integer,
  or (b) integerize `heat_dep` outright (cleaner, removes the last float on the heat gameplay path). Flag
  for Erik at that step.
- **Purely-visual float outputs stay float-OK (render-local):** `light_rgb`, `light_dx`/`light_dy`
  (`light_dir`), `smoke_glow` are render-only, may be one tick stale, produced/consumed on-device via
  CUDA-GL interop (§2.2 A), and are **exempt from P1** (gate `heat` only).

Runs once/frame (not per-substep), so its throughput weight is lower than the per-substep solvers — but
its *determinism* weight is full (it feeds unit HP), and it is the cleanest parallel win after the first
stencil, so it lands **early** (§7.4 S2), not last.

---

## 4. BUILD / TOOLCHAIN (multi-machine Ampere + Ada, VS2022)

### 4.0 What is on disk now (audited) + the LOCKED toolchain (Q1/Q2, 2026-06-27)

| Fact | Value |
|---|---|
| CUDA toolkit installed | **v12.4** (`nvcc` V12.4.131; the spike built with it) — **the locked toolkit (Q1)** |
| GPU / driver (desktop) | RTX 3070, **driver 610.62** (raised by Erik — lifts the CUDA ceiling; toolkit stays 12.4), sm_86 |
| VS2022 MSVC toolset | **VS2022 17.14** (MSVC newer than CUDA 12.4 officially lists) |
| `cpp/build/` cache | **"Visual Studio 18 2026" preview, MSVC 14.50 — POISONED for CUDA** (reset, Q2) |
| `cpp/build_vs2022/` cache | "Visual Studio 17 2022", x64 — proves VS2022 configures cleanly |
| `.pyd` | `breach_physics.cp311-win_amd64.pyd` |
| CuPy | `cupy-cuda12x` 14.1.1 on `numpy` 2.4.6, **installed + working** (Q3; suite 369 green) |

**The locked toolchain (Q1):** **CUDA 12.4 (already installed) + `-allow-unsupported-compiler`.** VS2022
17.14 is newer than 12.4's `host_config.h` lists, tripping nvcc's `#error -- unsupported Microsoft Visual
Studio version!`; the flag suppresses it, and for **pure-integer kernels** the "may cause incorrect
run-time execution" caveat is harmless (the bit-identity gate catches any codegen difference). Erik raised
the **driver to 610.62** (lifts the CUDA ceiling), but we **stay on the 12.4 toolkit** — no 12.6/12.8/12.9
download. Same setup on the Lenovo (Ada) later. *(The earlier "update driver + install 12.9 vs 12.6 U2"
framing is retired.)*

**The poisoned build dir (Q2) — TARGETED reset, not `rm -rf`.** `cpp/build/` is configured against the
**"VS18 2026" preview** (MSVC 14.50), which CUDA does **not** support. **Reset it onto VS2022 by deleting
the single `CMakeCache.txt` (+ the `CMakeFiles/` config dir) and re-running `cmake … -G "Visual Studio 17
2022"`** — a surgical reset that respects the deny-list (NOT a recursive force-delete of the tree). Also
**prune the stale `C:/tmp` worktrees.** A fresh `cpp/build_cuda/` is the equivalent clean alternative.

### 4.1 The compatibility matrix (resolved)

**Architecture support — settled.** Every CUDA 12.x toolkit (12.4–12.9) compiles both `sm_86` (Ampere)
and `sm_89` (Ada) — long-standing targets. A single fatbin with `sm_86` + `sm_89` SASS (+ one PTX for
forward-compat) is the intended multi-machine deployment: **one `.pyd` runs on both machines.**

**The MSVC ceiling (the only wrinkle) — RESOLVED by the flag (Q1).** The installed VS2022 17.14 MSVC is
newer than CUDA 12.4's `host_config.h` was validated against; nvcc hardcodes a max `_MSC_VER` and emits
`#error -- unsupported Microsoft Visual Studio version!` for a newer `cl.exe`. **The locked resolution is
`-allow-unsupported-compiler`** (suppresses the `#error`). NVIDIA's "may cause compilation failure or
incorrect run-time execution" caveat is **harmless here for two reasons**: (1) every synced kernel is
**pure integer** — two's-complement `+ − * >>` and integer atomics have PTX-defined, MSVC-independent
semantics, so there is nothing for an "unsupported" host compiler to mis-codegen; and (2) the bit-identity
gate (§1, §7) would catch any difference if there were. The driver is at **610.62** (lifts the ceiling),
the toolkit stays **12.4** — no toolkit download, no driver-vs-toolkit table to navigate. *(The old
two-path "12.9 vs 12.6 U2" analysis is retired; keep `-T version=14.40` pinning only in the back pocket if
the flag ever misbehaves.)*

### 4.2 The setup checklist (everything already in place)

1. **NVIDIA driver — 610.62, already installed** (Erik raised it; lifts the CUDA ceiling). Same line
   covers the Ada laptop later.
2. **CUDA Toolkit 12.4 — already installed** (`nvcc` V12.4.131; the spike built with it). **No download.**
   Build with `-allow-unsupported-compiler` (Q1).
3. **Nsight Compute + Nsight Systems + Nsight VSE** — bundled with the installed 12.4 toolkit.
4. (Have) Visual Studio 2022 17.14 — keep.
5. (Have) CMake ≥ 3.18 (`cmake_minimum_required`; CUDA-architecture support landed in 3.18). CMake
   ≥ 3.24 gives `CMAKE_CUDA_ARCHITECTURES=native` — optional.
6. (Have) **CuPy** `cupy-cuda12x` 14.1.1 on `numpy` 2.4.6 — installed + working (Q3; suite 369 green).

**Lenovo laptop (Ada, in a few weeks):** identical setup — CUDA 12.4 + the flag, the same fatbin (it
carries `sm_89`); a rebuild there just regenerates the identical `.pyd`.

### 4.3 CMake integration (against the real `cpp/CMakeLists.txt`)

Today: `project(breach_physics LANGUAGES CXX)`, `pybind11_add_module(breach_physics …8 .cpp…)`,
per-source `/fp:strict` on the six sim TUs, global `/O2 /fp:fast /arch:AVX2`,
`.pyd → <build>/Release/breach_physics.cp311-win_amd64.pyd`. Incremental CUDA changes (keep the CPU
build working alongside):

1. **Enable CUDA** (optionally behind a flag during bring-up):
   ```cmake
   project(breach_physics LANGUAGES CXX CUDA)   # or enable_language(CUDA)
   ```
2. **Dual architecture (one fatbin, sm_86 + sm_89 + PTX):**
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 75 86 89)   # 75 opportunistic (Turing data point)
   set(CMAKE_CUDA_STANDARD 20)              # the signed-shift portability floor
   ```
   CMake ≥3.18 turns `86;89` into `-gencode=arch=compute_86,code=sm_86
   -gencode=arch=compute_89,code=sm_89`. Append `89-virtual` (`code=compute_89`) to embed forward-compat
   PTX.
3. **Add `.cu` files to the existing pybind module** (they link into the same target):
   ```cmake
   pybind11_add_module(breach_physics  ...existing .cpp...  src/physics_engine.cu)
   set_target_properties(breach_physics PROPERTIES
       CUDA_ARCHITECTURES "75;86;89"
       CUDA_SEPARABLE_COMPILATION ON)
   find_package(CUDAToolkit REQUIRED)
   target_link_libraries(breach_physics PRIVATE CUDA::cudart)
   ```
4. **Flag hygiene.** Keep the MSVC-host-only `/fp:strict` on the `.cpp` TUs. For `.cu` TUs pass host
   `/fp:strict` through nvcc with `-Xcompiler=/fp:strict`, and the device side with
   `--fmad=false -prec-div=true -prec-sqrt=true -ftz=false` (and **never** `--use_fast_math`). The
   shipped lesson — "pin one truncation path cross-toolchain; `/fp:strict` is the determinism floor" —
   carries over.
5. **Host-compiler selection (the VS2022 requirement):**
   ```
   cmake -S cpp -B cpp/build_cuda -G "Visual Studio 17 2022" -A x64 \
         -DCMAKE_CUDA_ARCHITECTURES="75;86;89" \
         -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
   cmake --build cpp/build_cuda --config Release
   ```
   (`-allow-unsupported-compiler` is the locked Q1 flag — VS2022 17.14 is newer than CUDA 12.4 lists;
   harmless for the pure-integer kernels. `-T version=14.40` pinning is the back-pocket alternative if the
   flag ever misbehaves.) **Reset the poisoned `cpp/build/` first** (delete its `CMakeCache.txt` +
   reconfigure on VS2022 — the targeted Q2 reset, not a recursive delete) — or build in a fresh
   `cpp/build_cuda/`.
6. **The `.pyd` name/location is unchanged** — the CUDA port is invisible to `physics_runner.py`
   (`PhysicsRunner(breach_physics)` still imports the same module).

**Where the kernels plug in:** `PhysicsEngine` is the documented CUDA plug-in point; the solvers
deliberately re-fetch the raw numpy pointer each `step()` (the residency seam where a host pointer
becomes a device pointer). The unification work already landed the orchestration into `PhysicsEngine::
step*`, so kernels replace solver-method bodies one at a time.

### 4.4 Cross-arch compile from day one

`-gencode` both archs (75/86/89) in CMake so a kernel that *compiles* Ampere-only never merges (the old
`spike0/_arch_check.bat` habit, now a CMake property). The runtime cross-arch *digest* gate is §1.5 / §7.

---

## 5. The CONFIG-TOML NO-RECOMPILE preservation design (Erik's explicit requirement)

### 5.1 Today's result — verified: changing a `config.toml` value needs NO recompile

Traced end-to-end: `config.toml` (plain data) → `config.py` `tomllib.load()` at runtime (`CFG.reload()`
on F5/Ctrl+R) → `physics_runner.py` assigns `CFG.physics.*` onto the C++ solver instances as plain
attributes (`self.atmos.c = float(CFG.physics.wave_c)`, `self.fire.params.k_grow = …`) → `bindings.cpp`
exposes each as a pybind **`def_readwrite`** member (a `float`/`int` data member set at runtime). **There
is no `#define`/`constexpr` carrying any config value.** Quantization happens at the boundary, at load,
in float, once — never baked: scalar dials stay `float` members the solver `quantize()`s once per step;
per-tile material/thermal tables are baked **once at level load in Python** into **integer arrays**
(`heat_inv_shift`, `face_shift`, `ignition_temp_q16`), and the runtime kernel only indexes-and-shifts.
**Conclusion: every tunable is `config.toml` → `tomllib` → `def_readwrite` member or a load-time-baked
integer array. Zero constants are compile-time.** The few C++/`fixed_point.h` "constants" (`FP_SHIFT=16`,
`FP_ONE=65536`, `TEMP_SCALE`/`HEAT_SCALE`) are *format* invariants — the Q16.16 scale itself — and
correctly stay `constexpr`.

### 5.2 How the CUDA port preserves it

**The invariant:** anything a kernel reads per-cell that comes from `config.toml` must be a **runtime
value uploaded to the device, never a `#define`/`constexpr`/`-D` baked into the kernel.** Map every
config value to one of three device homes by access pattern:

| Config category | Today (CPU) | Device home (CUDA) | Why |
|---|---|---|---|
| Per-solver scalar dials (`wave_c`, `d_atm`, `k_grow`, `cool_shift`, water `k_p`… the `def_readwrite` members) | param struct members | **`__constant__` struct** per solver (`AtmosConst`, `FireConst`, …) **or** a small `__grid_constant__` kernel-param (CUDA 12.1+) | small, read-only, all threads read the same → textbook constant-memory broadcast |
| Per-tile **integer** tables baked at load (`heat_inv_shift`, `face_shift`, `ignition_temp_q16`, material id→property) | numpy int32 arrays | **device global memory** (`cudaMalloc` + `cudaMemcpy`), indexed as today | too big for constant memory; per-cell varying; already integer |
| Small material lookup tables (≤ dozens of ids) | numpy arrays | `__constant__` if access is broadcast-y, else global — **profile** | constant cache wins only on uniform access; divergent per-id reads serialize |
| Format invariants (`FP_SHIFT`, `FP_ONE`, `TEMP_SCALE`) | `constexpr` in `fixed_point.h` | **stays `constexpr`** (compiles into `__device__` unchanged) | not a tunable; baking it is correct |

**Rule of thumb:** if a value comes from `config.toml`, it arrives as a kernel argument or a `__constant__`
symbol — **never** as `#define X 4.0` or `constexpr float k_grow = 4.0` inside a `.cu`.

### 5.3 The upload-on-config-change mechanism

Preserve the "set attribute → it just works" UX:
1. **Keep the Python binding surface identical** — `physics_runner.py` still does `self.fire.params.k_grow
   = …`. The pybind setter now (a) stores the host value and (b) marks the solver's constant block
   **dirty**.
2. **Lazy upload at the top of `PhysicsEngine::step()`:** if a solver's const block is dirty,
   `cudaMemcpyToSymbol(d_fireConst, &h_fireConst, …)` once, then clear the flag. A config edit + the
   existing F5/Ctrl+R reload re-runs the binds, flips dirty bits, and the next `step()` re-uploads — **no
   recompile, mirroring today's restart/F5 behavior exactly.**
3. **Per-tile integer tables** re-upload only on level load / `on_tile_changed` (the existing CPU
   cadence), via a dirty-region `cudaMemcpy`.
4. **`__constant__` caveat:** `cudaMemcpyToSymbol` is not legal mid-kernel and isn't free — do it between
   launches (the top-of-step dirty check guarantees that). Config dials change rarely (on edit), so
   `__constant__` is the right home (read-cached, broadcast to all threads, no per-launch marshaling).

**Net: the no-recompile property is preserved by construction** — config values live in device runtime
memory (`__constant__`/kernel args for scalars, global memory for per-tile tables), uploaded on the
existing config-reload cadence, never compiled in.

---

## 6. CUDA BEST-PRACTICES tailored to OUR kernels

**0. Integer Q16.16 is what makes this port deterministic — keep the reductions integer, never
float-atomic.** NVIDIA's guidance is blunt: float atomics are "entirely disabled [for determinism]
because their non-associative semantics can yield nondeterministic results." Breach already won this:
`mean_sum`, `max_fire`, `max_wind_sq` port to integer `atomicAdd`/`atomicMax` or a CUB integer reduction
and stay bit-identical, where the retired float versions jittered (spike-0a). And the cost of
determinism here is **nil** — "there is no reason to calculate a parallel sum using nondeterministic
atomicAdd… the performance benefit is marginal at best." The correct choice is also the fast one.

**1. Memory coalescing — the row-major int32 layout is already optimal; protect it.** `Grid2D<T>` is
contiguous row-major, indexed `y*w+x`; one thread per cell with `x` fastest → a warp reads 32 adjacent
int32 words → four coalesced 32-byte transactions, the ideal. `cudaMalloc` is ≥256-byte aligned; each
row is `w*4` bytes (clean). The **stencil halo** is the watch item: N/S neighbours are a different row →
separate transaction streams (self + up-row + down-row) → ~3× traffic; that is what shared-mem tiling
(point 5) fixes. **Avoid strided layouts: re-plane `face_shift` (i*4, stride-4, 75% wasted) and
`light_atten`/`dyn_light_atten` (h,w,3) into separate coalesced planes.** The gas (N,h,w) array is
already plane-major — keep it.

**2. Occupancy — small blocks, watch the register-heavy kernels.** Default **256 threads as 16×16 or
32×8** (multiples of warp size; keeps x-coalescing). On cc 8.6/8.9 an SM holds 1536 threads → up to 6
resident 256-thread blocks if registers/shared-mem allow. The register cliff is in the **branchy**
kernels: `backtrace_sample_q` (DDA march + bilinear + Newton reciprocal) and the fixed-iteration
transcendentals (`reciprocal_q16` 4 iters, `sqrt_q16` 32 iters) are register-heavy — profile with
`nvcc -Xptxas -v` / Nsight Launch Statistics; if occupancy-bound, cap with `__launch_bounds__(256)`. The
transcendentals are register-heavy but **branch-identical → cost registers, not divergence** (the right
trade vs a libm `sqrtf`/`expf` that would diverge *and* threaten cross-arch bit-identity).

**3. Grid-stride loops — the standard kernel shape.** `for (int i = …; i < n; i += gridDim.x*blockDim.x)`.
Three payoffs: any grid size (240×480 → 1000×1000, no relaunch logic); thread reuse / occupancy tuning;
and **the determinism-debugging lever you will actually use** — launch `<<<1,1>>>` and the GPU kernel
must produce **bit-identical** Q16.16 output to the CPU solver. That is the golden-test harness for the
port (reproduce `tests/_s2_golden.pkl` / `_s4*_golden.pkl` on the GPU).

**4. Warp divergence — triage by *how* branches diverge.** Mask branches that are uniform across regions
(`if (!solid[i]) continue;`, the wall/vacuum skips) are **cheap** — walls/air cluster spatially, so a
16×16 tile is usually all-interior or all-wall; only boundary warps pay; **keep the branch**, the
compiler predicates short bodies automatically. The **global early-outs** (`max_fire` scan,
`water_any`/`ripple_any`) are CPU-serial `break`-on-first-nonzero patterns that **do not belong in a
kernel** — replace with an integer reduction + a **host-side launch decision**. The **genuinely
divergent** kernels are the DDA marches (smoke back-trace, raycaster) — mitigate with predicated corners
+ the physically-bounded ≤1–2 cell march (smoke) and ray-direction binning (raycaster).

**5. Shared-memory tiling — for the stencils and RB-GS.** Load a `(blockDim.y+2)×(blockDim.x+2)` halo
tile into shared memory once, `__syncthreads()`, read 4 neighbours from shared (~100× global bandwidth).
Apply to conduction, the wave/smoke Laplacians, and the RB-GS sweep. **Pad the tile to `TILE+1` columns**
to kill column-read (N/S) bank conflicts. RB-GS subtlety: color boundaries are **grid-wide syncs = one
kernel (or graph node) per color per sweep** — this preserves the "red reads only current black"
order-independence that makes it deterministic. **This is a profile-gated optimization, not a
correctness requirement** — at ship grid sizes the field fits in L2; land correctness first, tile the
bandwidth-bound stencils second.

**6. Kernel-launch overhead + CUDA Graphs — the substep loops are the poster child.** `run_substeps` is
a nest of small launches: `n_wave` × ~7 wave kernels + `diffuse_solve` (Dinv + `gs_iters×2` GS + residual
+ 2 BFS + wind ≈ 10–20) + `n_smoke × n_gases` × ~4 smoke kernels + `K × n_gases` sink-hops ≈ **100–300
launches/tick.** Each launch is ~2–5 µs; at ~150×4 µs ≈ 0.6 ms/tick of pure CPU-bound launch latency —
on a small grid that can *exceed* the compute time and scales with the training-farm instance count.
**Capture the per-tick solver chain as one CUDA Graph** (batch the setup once, replay as a single
submission). The fit is excellent: the per-tick *structure* is static; only the *counts*
(`n`,`n_wave`,`n_smoke`,`K`) vary → **cache a graph per `(n_wave,n_smoke,K)` tuple** (a handful of
discrete integer-cliff values). The counts are known at the top of the tick (exactly when you pick the
graph). **Determinism is unaffected** — a graph is the same kernels in the same order. The `.any()`
skip-empty-plane host branches become always-launch-but-cheap kernels (an all-zero plane through an
integer Laplacian is near-free) or CUDA 12.4+ conditional graph nodes; **start with always-launch.**
**The in-place buffer discipline (engine/02) is now load-bearing for graph validity** — a reassigned
field pointer invalidates a captured graph.

**7. The Nsight profiling workflow — what to measure, in order.** (1) **Nsight Systems first** — confirm
the launch-overhead diagnosis (the sawtooth of 4 µs launches with idle-GPU gaps → CUDA Graphs is the top
win; measure tick wall-time before/after). (2) **Nsight Compute on the heavy kernels** (the GS sweep,
`backtrace_sample_q`, conduction, the raycaster): read **global load efficiency / sectors-per-request**
(the coalescing + N/S re-fetch — <100% on a stencil = the shared-mem tiling signal), **achieved vs
theoretical occupancy + registers/thread** (the `__launch_bounds__` check), and **branch efficiency**
(quantify the DDA-march divergence). Use `compute-sanitizer` for race/uninitialized checks on the
in-place buffers. (3) **The determinism gate runs in parallel with perf, not after** — after every
optimization re-run the `<<<1,1>>>` golden test **and** a full-grid GPU-vs-CPU A/B; a tiling or
block-size change must be **0-ULP.** Profile-driven perf changes are the highest-risk place to silently
break the bit-identity you just shipped; gate every one.

---

## 7. SEQUENCED SUB-STEPS (Bedrock patch → the CUDA kernels)

Ordered by **algorithmic complexity × coupling** — leaf stencils first, the reduction-coupled RB-GS
diffuse last — and pedagogically monotone (one new GPU concept per step). **Every CUDA step keeps the CPU
method as the live fallback** (the `PHYSICS_BACKEND_<solver>` switch) so a half-ported engine is just
"some solvers on GPU, the rest on CPU," each gated. The game runs at every step.

**The shape after the decision walk (locked 2026-06-27):** a **pre-CUDA Bedrock patch** (CPU-only,
finishes the integer foundation) lands FIRST; then the CUDA arc begins with the first **kernel**
(temperature), **not** a cliff patch. The raycaster is an **early** kernel (it has a scatter wrinkle, so
it follows temperature once the plumbing is trusted). The diffuse RB-GS is last among the solvers. The
`mean_wp` **stopgap** ports with atmosphere; the CUDA-graphs pass is one dedicated step at the end; and
the `mean_wp` **edge-flux retirement** is an explicit post-arc milestone (§7.7).

### 7.0a The BEDROCK PATCH (pre-CUDA, CPU-only) — first & likely last

**Before any GPU kernel:** integerize the atmosphere/wave `n` and `n_smoke` substep-count cliffs like
water already is (§1.4). Convert atmosphere/smoke `max_dt()`→Q16.16, `d_eff_max`/`dt_stable`→integer, and
compute `n` via `fixedpoint::ceil_div`; **regenerate the affected golden; Erik feel-check.** This is
**not** a CUDA step — it is the last piece of the fixed-point bedrock (water proves the `ceil_div`
pattern; this brings atmosphere/wave/smoke to the same integer cliff). The cliffs are
double-but-correctly-rounded-deterministic *today* (not a live desync), so this is *completion*, not a
bug-fix — but doing it now means the GPU launcher inherits an already-integer cliff and there is no
`double std::ceil` left in `physics_engine.cpp`'s synced control flow. **Gate:** regenerated golden green;
no `double`-CFL cliff remains; Erik's feel-check passes. It lands once, before S-Temp, and is "first &
likely last" of the bedrock work.

### 7.0 Immediate pre-CUDA action — close the spike's cross-arch leg

Spike-0 passed on Ampere but its cross-arch leg **never ran at runtime** (`_arch_check.bat` only
*compiles* sm_75/sm_89). The `.cu` sources are missing from `spike0/` (only `.lib/.exp/.exe` and
`_runlog.txt` survive; they live on branch `origin/spike0-gpu-derisk`). **Recover the `.cu` sources,
then when the Lenovo lands, run `spike0a`/`spike0b` on Ada (and Turing), diffing
`0a_integer raw_int64 = -1514247643326` and `0b_integer = 0xAB27B2370160FFF4` against the Ampere
digests.** If those two integers reproduce byte-for-byte on Ada, the load-bearing premise is proven
cross-arch on Erik's own hardware **before a single production kernel is written** — the entire point of
a spike. This becomes the seed of the permanent X-ARCH rig (§7's gate).

### 7.1 The bit-identity harness (the P1 gate — extend `tests/field_ab_harness.py`)

Add (do not replace): a **canonical per-tick field digest** (blake2b over the concatenated int32/int16
field bytes in a **committed, versioned serialization `.toml`** — field→index→dtype→shape→endianness,
frozen before the first golden; int fields only, render floats excluded); a **`PHYSICS_BACKEND`** env
switch (`cpu`|`cuda`) read inside `PhysicsEngine`, **per-solver** (`PHYSICS_BACKEND_WATER=cuda`, rest
cpu) so each kernel is gated against CPU-int *in isolation*; **scratch hygiene** (`cudaMalloc` is **not
zeroed** — every kernel zero-fills or fully writes its scratch/halo; debug builds poison with a sentinel
and assert none survives — a *new* failure class the CPU arc never had); and the existing per-cell
divergence locator as the fallback on a digest mismatch. **Reuse the shipped S3c unit-state digest**
(`tests/test_s3c_unit_state_digest.py` / `field_ab_harness.SYNCED_UNIT_FIELDS` / `_capture_unit_state`):
it hashes per-unit HP/life/faction/position/footprint + the hit/kill **event stream**, so the GPU port
inherits the same fire→heat→kill watch on the Q2-fenced float-HP path — run it alongside the field digest
under `PHYSICS_BACKEND=cuda` (the field digest alone is blind to unit HP/life and the event stream).
**The shipped P1/P2 CPU tests stay green throughout** (the port changes *where* the int math runs, not
the math; the float-HP combat path is unchanged by the field port, still digested by S3c).

**Gate P1-GPU:** kernel K passes iff, for the full dangerous-path scenario (firestorm + flood + blast +
breach-mid-trajectory), `digest(K on CUDA) == digest(K on CPU)` every tick, `tol=0.0`.

### 7.2 The cross-arch gate (the only real cross-GPU proof)

P1 on one machine proves "CUDA-int == CPU-int *here*", not cross-GPU. **Gate X-ARCH:** the per-tick
digest from the **Ampere RTX 3070 (sm_86)** and the **Ada Lenovo (sm_89)** must be byte-identical to each
other and to the CPU-int golden. A committed runner (`tests/xarch_digest.py`) writes
`digest_<host>_<arch>.txt`; a `compare_digests` step diffs them. **Timing** (Erik owns the hardware): two
beats per kernel — (a) at merge, Ampere-vs-CPU P1 (immediate, blocks merge); (b) a **batched Ada replay**
when the Lenovo is online (re-run all merged kernels' scenarios on Ada, diff digests). **Maintain
`XARCH_PENDING.md`** listing every kernel merged Ampere-only; the Ada beat clears it. Until a kernel's
Ada row is green it is labelled *"single-machine determinism proven; cross-GPU UNVERIFIED."* Turing
(sm_75) is an opportunistic third data point (64-bit emulation behaviour). The desync canary (a CI job
that flips `--fmad=true`/`--use_fast_math` on a float TU and asserts the harness goes RED) proves the
gate can actually fail.

### 7.3 The first de-risk kernel — temperature, not RB-GS

Spike-0 already de-risked the *primitives* (reduction, reciprocal-GS) on Ampere. The first **production**
kernel should reach the moment-of-truth (CPU-int == GPU-int on Ampere AND Ada, *in the real engine*) as
cleanly as possible. **Temperature wins** head-to-head: a pure per-cell gather (no global reduction, no
per-cell divide, no conservation class, one pass), a leaf of the DAG (reads `heat`, writes `temperature`
— no float bridge to a not-yet-ported neighbour), the production template the whole arc was modelled on.
It exercises the entire **plumbing** path (upload caches, launch a trivial stencil, download, digest,
diff on two archs) with **zero algorithmic risk.** Leading with RB-GS instead would front-load the
reduction+barrier concept before the plumbing is trustworthy — and Spike-0b already proved the RB-GS
*arithmetic*. Lead with temperature; arrive at RB-GS late (S7).

### 7.4 The port order

**Pre-CUDA:** the **Bedrock patch** (§7.0a, CPU-only — integerize the atmosphere/wave/smoke cliffs) lands
first; then the spike cross-arch leg (§7.0). The CUDA steps below begin with the harness/toolchain, then
the first *kernel* (temperature).

| Step | Kernel(s) | New CUDA concept | Gate |
|---|---|---|---|
| **Bedrock** *(pre-CUDA, CPU-only — §7.0a)* | Integerize the atmosphere/wave `n` + `n_smoke` substep cliffs (`max_dt()`→Q16.16, `d_eff_max`/`dt_stable`→integer, `n` via `ceil_div` like water). Regenerate golden; Erik feel-check. **No GPU.** | — (finishes the integer foundation) | regenerated golden green; **no `double std::ceil` cliff left in `physics_engine.cpp`**; feel-check passes |
| **S0 — Toolchain + harness** | *No GPU physics.* CUDA 12.4 + `-allow-unsupported-compiler`/VS2022; `enable_language(CUDA)`; multi-gencode 75/86/89; annotate `fixed_point.h` `__host__ __device__`; build the per-tick digest + X-ARCH runner; hello-world memcpy + trivial-map kernel digested on Ampere. | **host/device model, `cudaMemcpy`, the toolkit compiles for device, the harness/X-ARCH rig** | hello-world digest identical CPU/Ampere; toolkit `static_assert`s pass on device |
| **S1 — Temperature** (`step_tail` conduct/cool) — *the first kernel* | **first real stencil; shared-memory halo tiling; scratch-zeroing** | P1 Ampere; X-ARCH pending→Ada |
| **S2 — Raycaster** (`cast_source_directional`) — *EARLY; the most parallel kernel, after temperature trusts the plumbing* (§3.8) | **per-ray parallelism + the SCATTER: multi-ray→cell heat deposit via integer `atomicAdd` (saturating); ray-direction binning for divergence; render floats via CUDA-GL interop** | P1 on the integer **`heat`** field only (render floats `light_rgb`/`dir`/`smoke_glow` exempt); **the heat float-then-quantize path held to `--fmad=false`/no-fast-math OR integerized** (§3.8); X-ARCH |
| **S3 — Water** (`step_water`: substeps, flow vx/vy, conserved depth) — re-derive the tilt slope in integer first (§3.1) | **multi-substep loop in one launch (ping-pong, no CPU round-trip); per-cell conservation on GPU** | P1 + **P2 conservation** Ampere; X-ARCH |
| **S4 — Smoke + 5 gas planes** — semi-Lagrangian advection (`run_substeps` gas loop) | **semi-Lagrangian gather, integer bilinear, the three rounding modes; int16 (Q1.15) packed [0,1] fields (the bandwidth win)** | P1 + P2 (per-plane mass) Ampere; X-ARCH |
| **S5 — Wave + wind + `mean_wp` STOPGAP** (`run_substeps` wave/transfer) — **port the int64-mean stopgap faithfully (Q5); the edge-flux retirement is the post-arc §7.7 milestone, NOT here** | **the hard concept: deterministic parallel reduction (int64 atomicAdd / warp tree) + the mid-tick grid barrier** | P1 (reduction-permutation + geometry sweep bites); **digest == the shipped CPU stopgap bit-for-bit** Ampere; X-ARCH |
| **S6 — Fire** (`step_tail` logistic) — work-list compaction, smoke-emit `atomicAdd`, burn-through compaction | **per-cell transcendental on device (`sqrt_q16`), divergent control flow, deterministic compaction** | P1 + the discrete extinguish/burn-through flip + **the device-vs-host `recip_mul` golden (per-cell 4× in the logistic, incl. NEGATIVE/signed inputs — §3.5)** Ampere; X-ARCH |
| **S7 — Atmosphere diffusion — integer RB-GS** (`run_substeps` diffuse) | **two-color GS as paired kernel launches; per-cell reciprocal at scale; roofline/occupancy profiling (Nsight)** | P1 + GS L∞ residual within a factor of the float build Ampere; X-ARCH |
| **S8 — `stamp_units` + persistent residency + CUDA graphs** — fields go GPU-resident (deltas up, per-tick full-field copy down), CUDA-GL interop, **graph capture (the single dedicated graphs pass, Q6)** | **GPU-resident state (the engine/02 seam); CUDA graphs (optimization, paid once, replayed — one pass after all solvers are bit-identical)** | P1 unchanged (residency + graph capture must not change a bit); end-to-end X-ARCH; perf measured |
| **(Post-arc, optional) — combat-kernel migration** (the Q2-lift, §2.2) | combat math → device kernels; integerize `dmg`/`current_hp` + tighten the S3c digest | P1 + the tightened unit-state digest |

**The `mean_wp` edge-flux retirement is a separate named post-arc milestone — §7.7 — not a step in this
table** (Q5: port the stopgap faithfully in S5, retire it after).

**Why this order:** the gating axis is determinism-provability. The Bedrock patch finishes the integer
cliff foundation off-GPU first. **Temperature is the first kernel** (a clean leaf stencil, the plumbing
de-risk, zero algorithmic risk). **The raycaster is early (S2)** — it is the most parallelizable kernel
(independent rays) and a gameplay-physics requirement (heat → damage), but it carries the one wrinkle
temperature lacks (the heat *scatter* + the float-then-quantize heat path), so it follows temperature
once the plumbing is trusted. The leaf solvers (water, advection) come next; the reduction (`mean_wp`
stopgap) is isolated into its own step (S5) because it is the one genuinely new *concept* and deserves a
clean gate; the diffuse RB-GS-with-a-barrier — the single hardest thing to prove bit-identical — is **last
among the solvers (S7)**; residency + the **one** CUDA-graphs pass close the arc (S8).

### 7.5 The learning notes (one concept per step)

- **Bedrock — finishing the integer cliffs.** Not a CUDA concept: `ceil_div` on Q16.16 for the
  atmosphere/wave/smoke substep counts (water already shows the pattern); the payoff is an integer cliff
  the GPU launcher inherits.
- **S0 — the host/device model.** Kernels, `<<<grid,block>>>`, `cudaMemcpy` up/down, the integer kit
  compiles to device PTX unchanged (the payoff of writing it portable). The mental model + the harness
  rig, zero physics risk.
- **S1 — the 2D stencil + shared-memory halo.** Temperature is the canonical "hello-world of GPU
  computing" (a 5-point stencil): thread-per-cell, halo-tiling, scratch-zeroing.
- **S2 — per-ray parallelism + the SCATTER (the raycaster).** Independent rays (the easiest parallel win)
  but a new hazard: many rays write one cell → integer `atomicAdd` (order-free, deterministic). The
  float-then-quantize heat path is the place to learn RULE B in anger (or integerize it). Render floats
  ride CUDA-GL interop and are gate-exempt.
- **S3 — multi-substep without a CPU round-trip.** Ping-ponging two buffers inside one launch so the CFL
  substeps never bounce to the CPU (the launch-overhead lesson by hand, before graphs).
- **S4 — gather (semi-Lagrangian) + 16-bit packing.** Advection is a *gather* (read interpolated source)
  → GPU-friendly; the int16/Q1.15 bandwidth win on the [0,1] fields (the half-rate-int datapath makes
  bandwidth the lever).
- **S5 — the reduction. THE hard concept.** Why a parallel sum is the deepest determinism hazard
  (spike-0a's float-atomic jitter is the lab demo) and why the *integer* int64 atomicAdd/shuffle-tree is
  order-free; plus the grid-wide barrier. Port the stopgap faithfully (the edge-flux retirement is §7.7).
- **S6 — per-cell transcendental + divergent control flow.** `sqrt_q16` per cell on device; warp
  divergence (the logistic's branch / the work-list compaction); a deterministic compaction (the
  burn-through list — control-flow output, not a field).
- **S7 — iterative solvers as paired kernel launches + profiling.** Red-black scheduling as two kernels
  with a barrier; per-cell reciprocal at scale; the **roofline/Nsight** loop (compute- or
  memory-bound?); the *deterministic*-vs-*converges* distinction (two claims, two tests).
- **S8 — GPU-resident state + CUDA graphs (the optimization).** The engine/02 deltas-up/per-tick-copy-down
  seam; CUDA graphs (capture the per-tick DAG once, replay with ~no launch overhead) as **one** dedicated
  pass. Deliberately **last** — you optimize a correct, resident pipeline, never a moving target.

### 7.6 The per-step gating contract (every kernel, no exceptions)

A kernel merges only when **all** hold: **(1) P1 Ampere** (`digest(CUDA)==digest(CPU-int)` every tick,
`tol=0.0`, dangerous-path scenario); **(2) P2** for conserved fields (Σfield constant to the LSB over a
long settle); **(3) scratch hygiene** (every buffer fully written/zeroed; sentinel-poison assert clean);
**(4) X-ARCH** (Ampere digest recorded; Ada beat clears `XARCH_PENDING.md`; Turing opportunistic);
**(5) CPU fallback intact** (`PHYSICS_BACKEND_<solver>=cpu` still passes the shipped harness — the game
runs with the kernel off); **(6) both archs compile** (multi-gencode, CMake-enforced); **(7)
golden/digest-spec versioned**, regenerated in the same commit (CI rejects a stale tag; the
reduction/GS steps additionally gate the **residual within a factor of the CPU build** —
*deterministic* and *converges* are separate claims); **(8) one kernel per commit** (`git revert` rolls
back exactly one solver's residency — no dual-path `#ifdef`).

### 7.7 POST-PORT MILESTONE — retire `mean_wp` for the local edge-flux (Q5, committed + tracked)

**This is an explicit named milestone, not a footnote** (Erik specifically asked it not be forgotten).
Through S5 the GPU runs the **stopgap**: the deterministic int64 global-mean reduction of `wave_p`,
ported bit-for-bit from the shipped CPU stopgap (`ac2cae8`). That global reduction is a **CPU-shaped
sync-barrier pattern** — it forces a grid-wide barrier mid-tick (wave→reduce→broadcast→subtract→diffuse).
**Once CUDA is up (after S5, ideally after the S8 residency/graphs pass), retire it for a local
edge-flux transfer** — order-free, barrier-free, no DC-bias, conservative by construction, GPU-native (no
reduction at all). This is the single biggest GPU-determinism + performance win remaining.

- **Why after the port, not during (Q5):** the port's gate is *bit-identity* to the CPU oracle. Changing
  the physics (reduction → edge-flux) **during** the port would break that gate — you could no longer tell
  a porting bug from the intended physics change. So port the stopgap faithfully first (S5), prove the GPU
  reproduces the CPU stopgap, **then** make the physics change as its own clearly-bounded step with its own
  before/after golden.
- **The deliverable:** the CPU edge-flux transfer (if not already shipped) + its GPU kernel, replacing the
  `mean_wp` reduction in both backends, with a fresh golden (the field *changes* here — this is a
  deliberate physics edit, regenerate and Erik-feel-check), and the mid-tick grid barrier removed from the
  graph. Track it in `XARCH_PENDING.md`'s sibling list until both archs are green.

---

## 8. DECISIONS (locked 2026-06-27 — formerly open questions)

All seven questions below were walked with Erik and are RESOLVED; the structural consequences are folded
into §0–§7 above. Retained here as the decision record.

**Q1 — Toolchain path. RESOLVED: the already-installed CUDA 12.4 + `-allow-unsupported-compiler`; stay
on the 12.4 toolkit (driver raised to 610.62, no toolkit download).** The earlier framing (driver-update
+ 12.9 vs 12.6 U2) is retired. VS2022 17.14 is newer than CUDA 12.4 officially lists, which trips nvcc's
`host_config.h` `#error`; `-allow-unsupported-compiler` suppresses it, and for **pure-integer kernels**
the "may cause incorrect run-time execution" caveat is harmless (the bit-identity gate would catch any
codegen difference anyway). Erik updated his **driver to 610.62** (which *lifts* the CUDA ceiling) but we
deliberately **stay on the installed 12.4 toolkit** — no 12.6/12.8/12.9 download. **Same setup on the
Lenovo (Ada) later.** (Folded into §4.)

**Q2 — Clean `cpp/build`. RESOLVED: yes — TARGETED reset onto VS2022 (delete `CMakeCache.txt` +
reconfigure), NOT a recursive `rm -rf`.** The VS18-2026/MSVC-14.50 cmake cache is poisoned for CUDA; reset
it to the VS2022 generator by deleting the single `CMakeCache.txt` (+ the `CMakeFiles/` config) and
re-running `cmake … -G "Visual Studio 17 2022"` — a surgical reset that **respects the deny-list** (no
recursive force-delete). Also **prune the stale `C:/tmp` worktrees**. (Folded into §4.0.)

**Q3 — Python device-array vehicle. RESOLVED: CuPy — chosen, INSTALLED, and WORKING.** `cupy-cuda12x`
14.1.1 on `numpy` 2.4.6; the **full Breach suite is 369 green** with it installed (numpy-2 is fine for
Breach). A `cupy.ndarray` gives `.data.ptr`, `__cuda_array_interface__`, and `cupy.asnumpy()` — the §2.4
device-backed-attribute contract for free. The env **coexists with the future PyTorch (ML)** stack, and
**CuPy↔PyTorch share GPU memory via `__cuda_array_interface__`** (zero-copy hand-off). (Folded into §2.4.)

**Q4 — Combat/gameplay field read. RESOLVED: copy ALL fields GPU→CPU each tick as the baseline, then
optimize per-system.** The integer fields ARE deterministic gameplay state, so gameplay reads broadly —
the **whole synced field set comes down once per tick** (~50 µs at Breach's 50×120 grid — effectively
free; correctness/access first). **NOT a subset-download:** transfer **latency** dominates, not bytes, so
batching the whole small field beats a fragmented per-element/per-subset gather; the "gather the specific
elements combat reads" idea is the wrong default (gather is only for huge sparse grids, which Breach is
not). **The optimization is per-system:** migrate a mature system's logic into a GPU kernel so it reads
on-device, shrinking the per-tick copy over time (the future Q2-lift for combat). (Folded into §2.0,
§2.2, §0.3.)

**Q5 — `mean_wp`. RESOLVED: port the deterministic STOPGAP (int64 global reduction) to GPU FIRST; the
edge-flux retirement is a COMMITTED, TRACKED post-port milestone with its own name (§7.7).** Porting the
stopgap bit-for-bit keeps the port a faithful CPU→GPU translation — a physics change *during* the port
would break the bit-identity gate. The edge-flux retirement (global reduction → local edge-flux; a
CPU-shaped sync-barrier pattern replaced by a GPU-native one) lands **after** CUDA is up, as the explicit
§7.7 milestone (not a footnote — Erik asked it not be forgotten). *(Distinct from the Bedrock cliff item:
Q5 is the `mean_wp` hot-path **sum**; the §1.4 Bedrock item is the atmosphere/wave `n` + `n_smoke`
**substep-count cliff arithmetic**. The Bedrock cliff integerization is a CPU-only pre-CUDA patch; Q5 is
a sequencing call — stopgap in S5, retirement in §7.7.)* (Folded into §1.3, §7.4 S5, §7.7.)

**Q6 — Persistent GPU residency + CUDA graphs. RESOLVED: one dedicated optimization pass AFTER all
solvers are ported + bit-identical (S8), not per-kernel.** Pure speed, correctness-first. Per-kernel
residency would re-plumb the host↔device boundary repeatedly while not-yet-ported solvers still need host
arrays, and graphs need a stable launch sequence. Keep the kernels on the simple "upload → launch →
download, gated against CPU" model until everything is on the GPU, then make it resident + graph-captured
once. (Folded into §6 item 6, §7.4 S8.)

**Q7 — Scope. RESOLVED (IMPORTANT CORRECTION): the RAYCASTER IS IN SCOPE — an early CUDA kernel, not a
later arc.** It is fixed-point and its heat rays **inflict damage** (the integer `heat` deposit →
`combat.apply_environmental_damage` → unit HP, verified in `combat.py:155-247` / `raycaster.cpp:218-220`),
so it is **deterministic gameplay physics, not render-cosmetic** — and the **most parallelizable kernel**
(independent rays). The earlier "render-side, later arc" framing (old §0.3/§7) is **changed**: bring the
raycaster in as an **early** kernel (S2, after temperature de-risks the toolchain, because the heat
DEPOSIT is a **scatter** → integer `atomicAdd`, deterministic, plus a float-then-quantize wrinkle — §3.8).
Gate the gameplay-affecting integer output (`heat`) on bit-identity; the purely-visual float outputs
(`light_rgb`/`light_dir`/`smoke_glow`) stay float-OK (render-local, CUDA-GL interop, gate-exempt). **Only
the combat-HP kernels remain separate** (the Q2-fenced HP math → the future Q2-lift). The arc ends when
the physics solvers **+ the raycaster** are on GPU + graph-optimized. (Folded into §0.2, §0.3, §3.8, §7.)

---

## 9. RISKS (ranked)

1. **`recip_mul` host-128-bit vs device path** (RULE A.2) — the one function where "same source" is not
   literal; the most likely single CPU≠GPU divergence until the `__mul64hi` device branch + the 10⁶-pair
   host↔device cross-check test are green.
2. **`cudaMalloc` is not zeroed → read-before-write of a padded/halo lane reads device garbage** the CPU
   harness is blind to. *Gate:* every scratch/halo buffer fully written or `cudaMemset`-zeroed each tick;
   debug poison-sentinel + assert-no-survivor (a **new** failure class).
3. **The toolchain wall** (VS2022 17.14 / MSVC newer than CUDA 12.4's `host_config.h` lists) — a hard
   gate on the *whole* arc. **RESOLVED (Q1):** CUDA 12.4 + `-allow-unsupported-compiler` (harmless for
   pure-integer kernels; the bit-identity gate catches any codegen difference); driver raised to 610.62,
   toolkit stays 12.4. Residual watch: confirm the flag'd build is bit-identical to the CPU oracle at S0
   (it will be — integer PTX is well-defined), before any kernel.
4. **The `dt` and tilt-slope host-`double` bakes** — CPU-only-deterministic today; cross-machine lockstep
   needs them proven bit-identical across the peers' host compilers (likely — scalar `+−×÷`, no
   FMA/transcendental) or integerized. *Gate:* add to the cross-machine digest; integerize (the
   tilt-slope re-derivation) if they diverge.
5. **The Ampere INT/FP datapath cost is a PERFORMANCE risk, not a determinism risk** — cc 8.6 has 16
   INT32 vs 32 FP32 cores/SM, so all-integer halves peak arithmetic throughput. It does **not** threaten
   bit-identity; it makes the int64-minimization (use the narrowest provably-safe intermediate), the
   no-divide reciprocal (shipped), the int16 [0,1] fields (locked), and shared-mem tiling load-bearing
   for *speed*. The roofline checkpoint is the diffuse RB-GS step (S7).
6. **CUDA-GL interop is a new correctness hazard** — the mapped-window mutual-exclusion rule ("accessing
   a mapped resource through OpenGL produces undefined results"). *Gate:* a tight RAII map/unmap wrapper
   so a stray raylib draw on a mapped texture can't fire.
7. **A future single-kernel grid-sync RB-GS reintroducing a cross-block RAW race** (RULE D). *Gate:* keep
   the two-launch form; digest-verify any grid-sync variant before adopting.
8. **Geometry-dependent indexing bug** (RULE E). *Gate:* the block-count sweep in the harness.
9. **CUDA-graph invalidation from a reassigned field pointer** — the in-place buffer discipline becomes a
   graph-validity rule. *Gate:* `compute-sanitizer` + the P1 digest after graph capture (S8).
10. **Launch overhead bottleneck at ship grid sizes** (~100–300 launches/tick) — a *throughput* risk
    addressed by CUDA graphs (S8), not a correctness risk.
11. **The raycaster heat float-then-quantize path** (now in scope, S2) — the heat deposit is computed in
    float (`heat_emit*heat_survival*dist_atten`) before the integer `heat_quantize`/scatter, unlike the
    pure-integer field solvers. *Gate:* hold the heat float path to `--fmad=false`/no-fast-math (RULE B)
    and gate the resulting integer `heat`, OR integerize `heat_dep` outright (cleaner — removes the last
    float on the heat gameplay path). The scatter itself is safe (integer `atomicAdd`, order-free); the
    risk is purely the float producing the quantized integer. Decide at S2.

---

## 10. REFERENCES

**NVIDIA primary:**
- [Floating Point and IEEE 754 (CUDA whitepaper)](https://docs.nvidia.com/cuda/archive/11.0/floating-point/index.html) — FMA single- vs two-rounding; ÷/√ correctly-rounded by default; serial-vs-parallel reduction results differ.
- [NVIDIA/CCCL Issue #5550 — Deterministic Algorithms (determinism taxonomy)](https://github.com/NVIDIA/cccl/issues/5550) — GPU-to-GPU / Run-to-run / Not-Guaranteed; float non-associativity + no order guarantee.
- [Controlling Floating-Point Determinism in NVIDIA CCCL](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) — float atomics disabled for determinism; integer/ordered reductions deterministic at marginal-to-zero cost.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) — coalescing (32-byte transactions, ≥256-byte cudaMalloc alignment), strided-access penalty, shared memory + bank conflicts, thread-block heuristics, pinned memory, minimize/batch transfers, constant memory, profiling/effective-bandwidth.
- [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) — `cudaHostAlloc`/`cudaMallocHost`, pageable-vs-pinned bandwidth, over-allocation caveat, batching.
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) — each output element written by one owning thread; geometry independence; the `<<<1,1>>>` serial-validation lever.
- [Finite Difference Methods in CUDA C/C++, Part 1](https://developer.nvidia.com/blog/finite-difference-methods-cuda-cc-part-1/) — tile+halo shared-memory stencil, `__syncthreads()`, constant-memory coefficient broadcast.
- [CUDA Programming Guide §Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html) — `cudaGraphicsGLRegisterBuffer`/`RegisterImage`, map/unmap-per-frame, register-once, the mapped-resource undefined-results rule.
- [CUDA Graph Best Practices — Introduction](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/introduction.html) — 2–5 µs per-launch latency, small-kernel idle-GPU failure mode, single-submission amortization.
- [CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) — `--fmad`, `-prec-div`, `-prec-sqrt`, `-ftz`, `--use_fast_math` implies `--fmad=true`.
- [CUDA Toolkit Release Notes — driver table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) · [CUDA Windows Installation Guide — VS2022/MSVC 193x host-compiler table](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) · [Ada Compatibility Guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/) · [Ampere Compatibility Guide](https://docs.nvidia.com/cuda/ampere-compatibility-guide/).
- [CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) · [Building CUDA Applications with CMake](https://developer.nvidia.com/blog/building-cuda-applications-cmake/) · [CUDA 12.1 large kernel parameters / `__grid_constant__`](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/).
- [nvcc unsupported-MSVC `#error` + `-allow-unsupported-compiler`](https://github.com/nerfstudio-project/nerfstudio/issues/3171) · [NVIDIA forum: _MSC_VER 1940 breaks builds](https://forums.developer.nvidia.com/t/msc-ver-is-1940-with-latest-vs-2022-upate-and-its-not-letting-me-build-llama-cpp/295748) · [CUDA/MSVC compatibility map](https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html).

**Literature:**
- [Impacts of floating-point non-associativity on reproducibility for HPC and deep learning (arXiv 2408.05148)](https://arxiv.org/html/2408.05148v3) — FP non-associativity as the root of GPU reduction non-determinism; integer arithmetic as the deterministic alternative.
- [Fast GPU algorithms for the red-black Gauss-Seidel method](https://www.researchgate.net/publication/271550237) — update-one-color-at-a-time, parity grouping, race-free parallel sweep.
- [Control-Flow Melding for SIMT Thread Divergence Reduction (arXiv 2107.05681)](https://arxiv.org/pdf/2107.05681) and [CUDA Thread Divergence](https://www.aussieai.com/blog/cuda-thread-divergence) — warp-divergence cost (up to 32×), predication and phase-splitting mitigations.
- [ACCU Overload 100 — Why Fixed Point Won't Cure Your Floating Point Blues](https://accu.org/journals/overload/22/124/) — fixed-point trades rounding/range for overflow/precision; what it buys is bit-reproducibility.

**Repo canon (verified in-tree):**
- `docs/fixed_point_migration_plan.md` (§1 determinism contract & blockers, §2 field table/overflow, §3 GS reciprocal + RB-GS + flux form, §4 reductions & cliffs, §6 harness/gating, §8.3 Ampere int datapath) — the plan this mirrors.
- `docs/s1_water_fixed_point_plan.md` / `docs/s2_fixed_point_plan.md` / `docs/s3_fixed_point_plan.md` — the shipped per-group plans.
- `docs/architecture/engine/02_state_and_ownership.md` — the GPU-residency / seam / freshness contract this realizes.
- `cpp/src/fixed_point.h` — the shipped GPU-clean toolkit (`mul_q16`/`mul_wide`/`narrow`, `reciprocal_q16` integer-Newton, `recip_mul` 128-bit host path, `mean_sum`/`mean_round`, `sqrt_q16` fixed-iter isqrt, `ceil_div`, `scale_mag`, `shr_round0`, `quantize`).
- `cpp/src/physics_engine.{h,cpp}` (the kernel seams: `run_substeps`/`step_water`/`step_tail`/`stamp_units`), `cpp/src/{water_solver,atmosphere_solver,smoke_dynamics,fire_simulation,temperature_solver,raycaster}.cpp`, `cpp/src/bindings.cpp` (`get_2d`, the `def_readwrite` param surface), `cpp/src/grid2d.h`.
- `cpp/CMakeLists.txt`, `cpp/build/CMakeCache.txt` (POISONED — VS18 2026/MSVC 14.50), `cpp/build_vs2022/CMakeCache.txt` (the VS2022 template).
- `config.py` / `config.toml`, `src/simulation/{physics_runner,gamemap,field_edit,recorder,simulation,combat,materials}.py`, `renderer/game_renderer.py`.
- `tests/field_ab_harness.py` (the P1 harness to extend), `tests/_s1_flux_truncation_check.py` (the host MSVC-vs-clang golden to mirror device-vs-host), `tests/_s2_golden.pkl` / `_s4*_golden.pkl`.
- `spike0/_runlog.txt` (the Ampere digests to match cross-arch: `0a_integer raw_int64 = -1514247643326`; `0b cpu_hash == gpu_hash == 0xAB27B2370160FFF4`; float-atomic VARIES; FMA fused 0xCEF16263 ≠ separate 0xCEF16261), `spike0/_arch_check.bat` (compile-only sm_75/sm_89 — the runtime cross-arch leg still owed); the `.cu` sources live on `origin/spike0-gpu-derisk`.
