# EOS P6 — GPU end-state alignment review

> **Status:** the scheduled between-P5-and-P6 review (decisions log, OPEN item "GPU END-STATE
> ALIGNMENT REVIEW"; Erik 2026-07-10: "the C++/CPU step is intermediate; step back and check
> alignment with the end goal"). Audits every primitive the EOS refactor introduced on the CPU
> (P1–P5.1, main @ `ce1f15a`) against the two true end states: **full GPU residency of the
> physics tick (S8)** and **batched many-environment RL training** (the engine is a state space
> for training agents — the project's real purpose). The CPU path is PERMANENT as the
> bit-identity reference, never throwaway.
>
> Doc-only review; no code was changed. Every quantitative claim is tagged
> **(counted)** = derived from the as-built code, **(measured)** = from
> `eos_p3_gate_measurements.md` / `eos_p3_microbench_results.md`, or **(estimate)** = labeled
> engineering estimate with its basis.

## Executive verdict

**P6 is GO — with adjustments, and one parked design decision that gates exactly one
sub-patch.** Every hot primitive the refactor introduced is either already in gather form
(single writer per cell — MG transfers, conduction, SL advection, compression work, the kick)
or two-color order-free (the RB-GS smoother), and every fixed-point helper on the new paths
has a proven device mirror from the cuda-breached arc or is FP_HD device-clean; the donor-cell
bulk flux is a near-line-for-line application of the shipped `cuda_water.cu` pattern. Two
adjustments are needed inside P6 authority: (a) the harness's `EOS_P6_PENDING` is one global
boolean and must become a per-kernel pending set before kernel-by-kernel unpinning can work at
all, and (b) the MG coarse tail should ship as a **fused single-block tail kernel**
(bit-identical; kills ~240 of ~400 naive launches/tick). The one genuine redesign flag: the
**P4/P5.1 combustion pass is sequentially order-dependent through non-solid flammables
(furniture)** — a mid-scan heat deposit can flip a later source's ignition gate, which no
parallel schedule can reproduce bit-identically; a small CPU-side gate-snapshot restructure
(PARKED §3.1, Erik's call, golden re-baseline once) unblocks its port. CUDA graphs and
env-batching are **S8 scope**, exactly as decisions #13 already assumes — P6's job is
correctness + digest proof per kernel, not speed; per-call P6 ports will NOT beat the CPU at
160²×1 env (transfer tax ~0.6–0.95 ms/call (measured, S8a spec) vs the whole MG solve at
3.2 ms CPU (measured)) and nobody should panic when P6 benchmarks say so.

---

## 0. The end-state numbers that frame everything (question A)

**MG pyramid at 160², as built** (`eos_solver.cpp` level loop, `mg_min_dim=1`, cap 9)
**(counted)**:

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| size | 160² | 80² | 40² | 20² | 10² | 5² | 3² | 2² | 1² |
| cells | 25,600 | 6,400 | 1,600 | 400 | 100 | 25 | 9 | 4 | 1 |

Total 34,139 cells ≈ 4/3 × fine (the textbook ratio).

**Naive launch tally, V(2,2)×C=2, coarsest 32 sweeps, one kernel per color per sweep
(counted):** down-leg 6 launches/level × 8 = 48; coarsest 32 sweeps × 2 colors = **64 launches
on ONE cell**; up-leg 5 × 8 = 40 → 152/cycle → **304/tick for the solve**. Levels with ≤1,024
cells (levels 3–8) receive **238** of those 304. Add the substep loop (≤8 substeps × ~7–11
kernels), level build (~10–18), p*/div/kick/compression (~5): **~380–450 launches/tick/env
naive**. At 3–10 µs/launch on Windows WDDM **(estimate; driver-dependent)** that is
~1.5–4 ms/tick of pure launch latency — the launch tail, not arithmetic, is the dominant GPU
cost at 160²×1 env. This is the same conclusion the decisions log reached ("~0.5 ms/tick
naive"), with the honest note that the as-built full pyramid + 32-sweep coarsest is worse than
that early estimate.

**Where the tail actually bites and what cures it:**
- **Fused coarse tail (P6 scope, mechanical — §2.2):** one kernel for levels 3–8 removes ~238
  launches/tick with zero digest risk. Naive count drops to ~170/tick.
- **CUDA graphs (S8b scope):** the V-cycle is a fixed launch shape (schedule frozen, level
  count fixed by grid size) — ideal for capture; the one data-dependent shape is
  `n_sub ∈ {1..8}` → capture 8 graph variants or use graph update. Graphs make the residual
  launch tail ~nil. Not P6: per-call ports with D2H digest gates between kernels have nothing
  useful to capture.
- **Batching B envs (S8/RL scope):** one launch serves B envs, so per-env launch overhead
  divides by B; occupancy inverts the problem — level 0 at B=64 is 1.6M threads (saturates any
  consumer GPU), and even the 3² level at B=256 is 2,304 threads. Batching structurally cures
  the coarse-level starvation the single-env case suffers. **(counted arithmetic; occupancy
  claim is standard-GPU-architecture estimate)**

**Batched solve cost sanity check (estimate):** smoother+residual traffic ≈ 76 B/cell-pass
(m, gE×2, gS×2, recip, b, 5×P), ~350k cell-passes/tick/env over the pyramid → ~27 MB/tick/env;
at B=256 ≈ 6.9 GB/tick → 7–14 ms on a 500–1000 GB/s card, i.e. **~30–55 µs per env** for the
pressure solve vs 3.2 ms on CPU (measured) — a ~60–100× per-env throughput gain, memory-bound.
This is the RL-training payoff the whole review exists to protect; nothing in the as-built
design forecloses it.

---

## 1. PORT AS-IS (confirmed)

Primitives where the CPU design is GPU-sound exactly as written. Evidence per item.

1. **RB-GS smoother two-color structure** (`eos_solver.cpp:717–753`). Within a color, every
   read is either an opposite-color neighbor (4-stencil: parity flips), the cell's own
   pre-update `P`, or per-cell constants (`m`, `g`, `recip`, `b`) — so within-color updates are
   **order-free**, and one kernel launch per color (with a barrier between colors, i.e. the
   launch boundary) reproduces the CPU's sequential sweep bit-for-bit **(counted from the
   stencil)**. This is the load-bearing fact of the whole port: the smoother needs no
   restructuring at all.
2. **Variational/Galerkin MG transfers are already pure gathers** (question D). Restriction =
   per-coarse-cell SUM of its ≤4 children's residuals (single writer, `restrict_res`);
   prolongation = per-fine-cell read of its ONE parent, write own cell (`prolong_correct`);
   coarse-level build = per-coarse-cell gathers of child masses/anchors and crossing-face
   conductance sums. **There is no scatter anywhere in the MG hierarchy** — the
   precompute-then-gather idiom is not even needed; determinism holds by single-writer
   construction. Access pattern: 2-strided child reads on restriction (mildly uncoalesced,
   fine at these sizes); prolongation's parent read is a 2×2 broadcast (cache-friendly).
3. **Donor-cell bulk flux ≙ `cuda_water.cu` K3–K8, near-mechanically** (question F).
   `bulk_flux_transport_cached`'s five stages (face flux → dq → outflow-limiter scale →
   scale-apply → divergence apply → clamp) are stage-for-stage the water kernels K3–K7(+K8),
   including the same `flux_to_dq` 128-bit truncation, the same `scale_mag` magnitude-first
   shrink, and the same gather-then-apply conservative form. Deltas are mechanical: per-face
   coefficient arrays instead of one scalar (K4's signature grows one pointer), 2 conservative
   planes (loop or fold into the thread index), and the all-zero-plane skip — which is **pure
   perf, arithmetically a no-op** (an all-zero plane produces all-zero fluxes and unchanged N),
   so the GPU may simply drop it or keep a host-side flag; either is bit-identical **(counted)**.
4. **Fused 3-field SL advection is a per-cell gather** — DDA march + bilinear over `cmask`,
   reads frozen `src` copies, writes own cell. Direct precedent: `cuda_smoke.cu` (S4a) ports
   exactly this backtrace class including `reciprocal_q16_dev`. The march's data-dependent step
   count causes warp divergence (perf, not determinism). The `cmask` table is a per-tick
   precompute — one trivial kernel.
5. **Per-cell single-writer kernels:** `p*` materialization, `div_u`, the momentum kick +
   absorption + clamp chain, compression work (4c), `P` store, `P_prev` copy, Dalton `n_total`
   sum, per-cell diagonal reciprocals, `coeffE/S` build, conduction Pass 2 (explicitly
   double-buffered "no scatter, no atomics" — `temperature_solver.cpp:281`), cooling Pass 3,
   heat-convert Pass 1 (per-cell deposit; the CAS-atomic variant in `cuda_raycaster.cu` is NOT
   needed here — single writer). All port as plain grid-stride kernels.
6. **The reductions are order-free integer maxes** (`max_rad`, `t_max_abs_raw`, `max_du_raw`) —
   the proven spike-0a/`mean_sum` class; a standard integer max tree/atomicMax gives the
   identical result on any lane order **(counted; the codebase's own documented property)**.
   `n_sub` derivation (`ceil_div`) stays host-side — 2–3 scalars D2H per tick pre-substep in
   the per-call era; a tiny sync point at S8 (noted, acceptable).
7. **Warm start is state** (`L.P[i] = p_prev[i]`): deterministic by definition; at S8 the
   `levels_` scratch becomes persistent device scratch (the S8a "C++-owned persistent scratch"
   pattern, already specified).
8. **Wide-int64 on device is affordable as-is; do NOT restructure** (question C). Inventory of
   the solve chain's wide ops **(counted)**: smoother/residual = 6× `mul128_shr` per cell-pass
   (1 mass + 4 faces + 1 recip-apply); level-0 build = 3–4 per cell + one int64 divide
   (`2^32/aK`); per-level diagonal = one int64 divide per cell (`2^48/d`, ~34k/env/tick);
   kick/CFL = ~4 per open cell + one Newton `reciprocal_q16` (int64-only body); K bridge =
   host-side scalar. On Ada/Ampere a 64×64→128 multiply emulates to ~6–8 32-bit IMADs
   **(estimate)** → ~350k cell-passes × ~40 IMAD ≈ 14M IMAD-equiv/env/tick ≈ **microseconds of
   arithmetic on a 4070-class part; the solve is memory-bound, not int64-bound** (§0 traffic
   estimate). 64-bit integer division is an emulated subroutine but exact and deterministic
   (C++ semantics) and the counts are tiny. Restructuring to fewer/narrower muls is both
   unnecessary and mostly impossible without digest changes: the per-level budgets show deep-
   level `g×ΔP` products reaching ~7×10¹⁸ **(counted from the ×2/level conductance-sum growth
   over 8 levels on the level-0 bound)** — at the int64 edge, which is exactly why the 128-bit
   staging exists. One legitimate container-only note: level-0 `g` values provably fit int32
   (bound 6.5×10⁷ raw at the N floor), so a per-level int32 storage split could halve
   coefficient traffic **with identical values** — an S8 memory-layout option, not P6.
9. **P5.1 `wall_hp` decrement is parallel-safe per tile** (question F, verified): only
   iteration `i` ever writes `wall_hp[i]` (`combustion.cpp:107–108`); the up-to-4 subtractions
   + 1-LSB floor happen in `i`'s own D4 order. Given the per-face burn values, a per-`i` thread
   replays the exact chain — single writer, deterministic. (Its *inputs* are the combustion
   pass's burns — see §2.3/§3.1.)
10. **Device-mirror coverage is essentially complete** (question E, audited against
    `cuda_fixedpoint_device.cuh` + `fixed_point.h` FP_HD annotations):

    | helper used by the new paths | device status |
    |---|---|
    | `mul128_shr` (shifts 8/16/40) | `mul128_shr_signed` ✓ (same hi:lo combine as the MSVC host path) |
    | `reciprocal_q16` | `reciprocal_q16_dev` ✓ — verified same as-built body (top-2-bit seed, 4 round-to-nearest Newton iters) |
    | `sqrt_q16` | `sqrt_q16_dev` ✓ (fixed 32-trip) |
    | `recip_mul` (c_v bridge) | `recip_mul_dev` ✓ |
    | `scale_mag` (limiter) | `scale_mag_dev` ✓ |
    | `flux_to_dq` (bulk flux) | `flux_to_dq_dev` ✓ but file-local in `cuda_water.cu` — **hoist to the shared .cuh (P6.1 item)** |
    | `mul_q16`/`mul_wide`/`narrow`/`narrow_round`/`sat_add_q16`/`shr_round0`/`ceil_div` | FP_HD pure-int64 — device-clean as-is ✓ |
    | int64 `/` (2^32/aK, 2^48/d, limiter, u_est) | native C++ integer division — exact, deterministic ✓ |
    | `heat_saturating_add` (combustion + T Pass 1) | **missing as a shared plain device helper** — `cuda_temperature.cu:37–50` inlines it and `cuda_raycaster.cu` has only the CAS-atomic variant; add the trivial non-atomic mirror to the .cuh (P6 work item) |
    | `quantize` (per-cell on `dyn_wave_absorb`, per-face on perm) | FP_HD and bit-safe on device (×2^16 is exact in double; cast rounding is fixed) — but prefer the §2.5 hoist |

11. **Retirements confirmed** (question G): `run_substeps` calls `this->eos.step(...)` only
    (`physics_engine.cpp:241`); the pre-P3 wave+diffuse path survives solely as a named,
    never-called reference function (`physics_engine.cpp:322–329`, scheduled for deletion).
    `cuda_wave.cu` / `cuda_atmosphere.cu` are reachable ONLY via their direct pybind test
    entries (`bindings.cpp:346ff`, `:455ff`) and the s5/s7 checks — all dead under
    `EOS_P6_PENDING`. **Delete in P6.0:** both `.cu`+`.h`, their CMake lines
    (`cpp/CMakeLists.txt:74,76`), their bindings, `tests/cuda_s5_check.py` /
    `cuda_s7_check.py` / `test_cuda_s5_wave.py` / `test_cuda_s7_diffuse.py`, and the
    `tools/run_on_cuda.py` backend-list entries. (The CPU `atmosphere_solver.*` deletion stays
    P7 cleanup, per the design's own deprecation rule.) Note the two stale-kernel guards that
    P6 must resolve rather than delete: the fire-plume assert (`physics_engine.cpp:133`) and
    the trace-smoke cadence assert (`physics_engine.cpp:280`).

---

## 2. MECHANICAL ADJUSTMENTS (within P6 authority — cannot alter any digest)

1. **Harness: per-kernel unpinning** (question H prerequisite). As built,
   `EOS_P6_PENDING = True` is a single module global (`tests/cuda_harness.py:63`) and
   `cuda_available()` returns False unconditionally while it is set — there is **no partial
   unpinning today**; the design's "flip back kernel by kernel" has no mechanism. Replace with
   a pending SET, e.g. `EOS_P6_PENDING = {"wave","atmosphere","smoke","fire","temperature",
   "water",...}` and `cuda_available(kernel: str)`; each gate names its kernel; each P6
   sub-patch removes exactly its own key (and P6.0 removes "wave"/"atmosphere" by deleting
   their gates). Pure test infrastructure — zero digest surface. Determinism argument: none
   needed (no sim code).
2. **Fused coarse-tail kernel** (question A's P6-scope answer). Run all levels with ≤~1,024
   cells (levels 3–8 at 160²) inside ONE kernel, one thread block, `__syncthreads()` standing
   in for every CPU pass boundary (color→color, sweep→sweep, smooth→residual→restrict→
   prolong). Removes ~238 of ~304 solve launches/tick **(counted)**, including the 64
   one-cell launches at the coarsest level. **Determinism argument:** the arithmetic sequence
   is unchanged — the only concurrency introduced is within a color, which item §1.1 shows is
   order-free; every cross-color/cross-stage dependency is separated by a block-wide barrier
   exactly where the CPU has a loop boundary. Bit-identical by construction, digest-gated like
   everything else.
3. **Combustion: the precompute-then-gather face-buffer split** (question F — the arithmetic-
   preserving half; the other half is PARKED §3.1). The CPU pass's neighbor writes
   (`O2[j] -= burn`, `SOOT/N2[j] +=`, `temperature[j]` deposit) are scatters with an
   order-dependent clamp chain — but the order is FIXED: for a given burn site `j`, the
   row-major scan applies its ≤4 sources in the order **from-N, from-W, from-E, from-S**
   **(counted: source rows y−1 < y < y+1; within row y, x−1 < x+1; each source touches j via
   exactly one direction)**. So: **pass A** — one thread per burn site `j` replays that fixed
   4-source chain locally (threshold gate → `min(cap, o2)` clamp → O2/SOOT/N2 updates → the
   post-burn-N heat deposit, all in-thread sequential), writing the per-face burn amounts to 4
   direction-keyed face buffers (one writer per element — the water `dq_e/dq_s` pattern);
   **pass B** — one thread per source `i` gathers its 4 face burns and replays the N,S,W,E
   `fuel_cost` subtract+floor chain on `wall_hp[i]` (§1.9). This reproduces the CPU interleave
   bit-exactly **provided the source gates (prefilter, `T ≥ ign`) read pass-entry state** —
   which today they do not (see §3.1). This adjustment is therefore contingent: it ships
   after (or together with) the parked CPU change.
4. **Kernel fusion in the substep loop:** src-copy + SL-advect + zero-u-on-solid fuse into one
   kernel (independent per-cell ops on frozen inputs); the bulk-flux stages fuse across the 2
   conservative planes (plane index in the thread id). Bit-identical: no dependency crosses the
   fused boundary that wasn't already satisfied within one cell's computation.
5. **Hoist the per-cell `quantize(dyn_wave_absorb[i])`** in the kick loop
   (`eos_solver.cpp:881`) into a per-tick precomputed q16 plane — same double math, same
   rounding, per-tick-constant input; identical class to the blessed P3 hoists ("absorb·dt
   quantize hoisted"). Same for the per-face `quantize(pf)` in the `gE/gS`/`coeffE/S` builds if
   the level build moves on-device (FP_HD `quantize` is bit-safe on device regardless — ×2^16
   is exact in double and the cast rounding is toolchain-fixed — the hoist is for cleanliness
   and float-free kernels, not correctness).
6. **Digest strategy for P6 gates: host-side, unchanged.** `digest_of` is an order-dependent
   sequential FNV (its own comment flags it non-portable to GPU) — but per-call P6 kernels
   return their buffers to host anyway, so every gate is a D2H + host digest / byte-compare
   (the established `cuda_s*_check` + `_xarch_perfield_digest` pattern). No device-side digest
   is needed until S8 wants an on-device gate mode; defer.
7. **Level build placement is a free choice pre-residency:** build levels on host and H2D them
   (simplest first cut) or port the ~10 small build kernels — both are digest-neutral;
   transfers dominate either way in the per-call era. Decide by implementation convenience in
   P6.3; the S8 endpoint is on-device build (all gathers, §1.2).
8. **Batching forward-compatibility (design kernels now, batch later):** index every kernel
   `(env, cell)`-ready — per-env scalar tables (`dt_s_q`, `c_local_q`, `n_sub`, `Kdt_raw`)
   instead of hard-coded scalars costs nothing now and makes S8-era batching a layout change
   instead of a rewrite. Per-env `n_sub` divergence at batch time = run `max(n_sub)` substeps
   with a per-env guard (a guarded-off substep must be an exact no-op — verify at S8b design
   time, flagged). Shared-map RL batches + env-innermost layout give warp-uniform branching.
   **(design note, zero P6 arithmetic impact)**

---

## 3. PARKED — DESIGN-CHANGE RECOMMENDATIONS (Erik's morning read)

Everything here changes arithmetic/results and is NOT P6's to decide.

1. **Combustion gate-snapshot restructure — RECOMMENDED BEFORE its P6 sub-patch (the one real
   redesign).** The as-built pass is Gauss-Seidel in a way no parallel schedule can reproduce:
   a source's heat deposit lands in open-air neighbor `j`, and a **non-solid flammable** tile
   is both a valid burn site `j` AND a source `i` — furniture qualifies (config:
   `materials.furniture` `flammable = true`, `permeability = 0.5`; `gamemap.py:409`:
   `solid = permeability <= 0`). So a deposit made early in the row-major scan can push a
   furniture tile over `ignition_temp` and flip its own gate later in the SAME pass — an
   unbounded down-and-right intra-pass ignition cascade **(counted from
   `combustion.cpp:69–133` + the config/mask derivation)**. **The fix:** the gates (prefilter
   and `T ≥ ign`) read a frozen pass-entry `T` snapshot; deposits, clamps, and the O2 chain are
   untouched. With that one change, §2.3's face-buffer split is exactly bit-identical to the
   CPU. **Behavioral delta:** a same-tick mid-scan ignition cascade defers one tick — combustion
   is once-per-tick cadence anyway (design §10.4 already lists cadence as revisitable), and the
   scan-order-dependent cascade is arguably an artifact, not physics: today a fire spreads
   through furniture faster down-right than up-left. **Cost of NOT doing it:** combustion stays
   CPU-only forever, or the GPU port abandons bit-identity (non-negotiable). **Re-verification:**
   unit tests + the P5.1 lifecycle E2E re-run + perturbation trio + golden re-baseline ONCE
   with rationale. Small patch; recommend doing it as the design-gate for P6.6.
2. **RB-GS → Chebyshev-Jacobi smoother swap — NOT worth it before P6; likely never.**
   Quantified: RB-GS on GPU costs 2 launches/sweep and either a parity-branch (half threads
   idle) or index-compacted half-grids (mechanical); Chebyshev-Jacobi is 1 launch/sweep,
   branch-free, comparably smoothing. But the swap changes every Helmholtz digest, **invalidates
   the MEASURED schedule freeze** (V(2,2)×C=2 warm-started was frozen from 300-tick durability
   data for THIS smoother; the project has twice been burned by "asymptotic ≠ fixed-schedule" —
   Chebyshev needs its own eigenvalue-bound estimation, stability sweep, and golden), and its
   entire benefit — launch count and divergence — is precisely what batching (+graphs) already
   amortizes by B at S8. RL-batching lens says: keep RB-GS. Revisit only if S8 profiling shows
   the smoother itself (not launches) dominating, which the §0 traffic estimate says it won't.
3. **Coarse-tail ALGORITHM change (truncate the pyramid / batched direct solve at ~10²–20²) —
   PARK indefinitely.** Any truncation or direct-solve substitution changes arithmetic and
   re-opens the MG measurement gate (the full-pyramid 1×1-coarsest schedule was frozen from
   data; C=1/V(1,1) were measured too marginal). The mechanical fused-tail kernel (§2.2)
   captures essentially the whole win (launch count) with zero digest risk, and batching
   removes the occupancy motivation. There is no remaining payoff to justify re-measurement.
4. **CUDA graphs — S8b, as already planned (decisions #13).** P6's per-call, gate-instrumented
   kernels have nothing useful to capture. Graph-shape note for S8b: `n_sub` is the only
   data-dependent launch shape (8 variants or graph-update); the MG cycle is fixed-shape and
   capture-friendly — a direct consequence of the "fixed schedule, never adaptive" determinism
   rule, which turns out to be exactly the CUDA-graphs-friendly property. The determinism
   discipline and the performance end-state are aligned, not in tension.
5. **Batching-across-envs — the architecture is compatible; the work is real but S8/RL-arc.**
   Per-env arithmetic is untouched by batching (every kernel is per-cell or per-color
   independent), so per-env bit-identity against the CPU reference survives — the digest gate
   generalizes to per-env digests. The genuinely new design work: per-env reduction outputs
   feeding per-env scalar tables, the guarded max-`n_sub` substep loop, and (for the
   training-throughput win) env-innermost memory layout on shared-map batches. Recommend a
   short S8b design-gate for the batching seam before the first big training run; P6 only
   needs §2.8's zero-cost forward-compat.

---

## 4. Recommended P6 sub-patch order + digest-gate plan (question H)

Ordering principle: infrastructure first, precedent-backed kernels next, the solver in the
middle once the gate machinery is proven on easy kernels, the parked-decision-dependent pass
last. Every sub-patch: build → per-kernel A/B vs CPU (byte-compare / host digest over a
40-tick two-run trajectory, the `cuda_s*_check` pattern) → cross-machine per-field digest
(Ada vs Ampere, the cuda-breached protocol) → remove its key from the pending set → suite
green.

| # | sub-patch | precedent / risk | digest gate | unpins |
|---|---|---|---|---|
| P6.0 | Harness pending-SET rework; **retire `cuda_wave.cu`/`cuda_atmosphere.cu`** (+bindings, CMake, s5/s7 checks+tests, `run_on_cuda.py` lists); shared-.cuh hoists (`flux_to_dq_dev`, plain `heat_saturating_add_dev`) | none (no sim code) | suite green; grep-no-callers | removes "wave"/"atmosphere" keys |
| P6.1 | Bulk donor-cell flux kernels (water K3–K8 pattern, per-face coeffs, 2 planes) | `cuda_water.cu` — lowest risk | `digest_bulk_flux` trajectory + per-plane byte-compare | key "bulk_flux" (kernel-gate only; engine dispatch waits for P6.5) |
| P6.2 | Fused 3-field SL advection + cmask build + zero-solid | `cuda_smoke.cu` backtrace class | `digest_advect` trajectory | key "advect" |
| P6.3 | MG pressure solve: level build (host or device), smoother, residual, transfers, **fused coarse tail (§2.2)** | the hard one; §1.1/§1.2 arguments | `digest_helmholtz` per tick + per-cycle level-`P` compares vs an instrumented CPU run + the §3.4 overflow stress sweep re-run on device | key "helmholtz" |
| P6.4 | Kick + absorption + clamp chain (+ §2.5 hoist); compression work | per-cell, trivial | `digest_velocity`, `digest_compression` | keys "velocity","compression" |
| P6.5 | EOS orchestration: p*, div_u, reductions, host `n_sub`; full `eos.step` per-call on GPU; engine dispatch + unpin | integration risk only | all six digests, 40-tick two-run determinism, cross-machine per-field | key "eos" (the big flip) |
| P6.6 | Unified conduction + Pass 1/Pass 3 (extend `cuda_temperature.cu`; n_bulk input) | gather stencils, low risk | temperature trajectory byte-compare | key "temperature" |
| P6.7 | Trace-smoke re-port at the new once-per-tick cadence (resolves the `physics_engine.cpp:280` assert) | `cuda_smoke.cu`, cadence re-derivation | s4a-class check re-derived | key "smoke" |
| P6.8 | Fire re-derivation (stale plume→T-shim + n_o2 signature — the `:133` assert) | `cuda_fire.cu` exists but stale | s6-class check re-derived | key "fire" |
| P6.9 | **Combustion** (face-buffer split §2.3) — **gated on the §3.1 parked decision landing first on CPU** | the redesigned pass | gas planes + T + wall_hp byte-compare + P5.1 lifecycle E2E + perturbation trio | key "combustion"; pending set now empty → delete `EOS_P6_PENDING` machinery |

Dependencies: P6.1–P6.4 are independent of each other (parallel worktrees fine, one branch
each per the house rule); P6.5 needs all of P6.1–P6.4; P6.6–P6.8 independent after P6.0;
P6.9 last, after its design-gate.

---

## 5. Honest residue

- The launch-overhead figures (§0) are estimates; the first P6.3 build should spend ten
  minutes with Nsight confirming the naive-vs-fused-tail launch cost before anyone tunes
  further.
- The §3.1 combustion coupling was established by code reading (furniture is the witness);
  a one-scenario CPU experiment (two adjacent furniture tiles straddling ignition, one heated
  by a scan-earlier source) would make the cascade concrete for Erik's decision — cheap to add
  to the P6.6 design-gate.
- `TemperatureSolver`'s GPU dispatch (`physics_engine.cpp:183`) still passes the OLD signature
  (no `n_bulk`) to the stale kernel — currently unreachable only because the backend flag
  defaults false AND the harness pins; P6.0's pending-set rework should also add a hard assert
  there (the fire/smoke asserts' sibling) so the migration-window rule ("stale kernels
  unreachable, not just unused") holds by construction, not by flag discipline.
