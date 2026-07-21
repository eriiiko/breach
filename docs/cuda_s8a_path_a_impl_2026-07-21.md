# CUDA-S8a Path A — EOS device residency, implementation design (2026-07-21)

**Status:** DESIGN v2 — v1 survived a 3-lens adversarial critique (Galerkin
bit-exactness / reduction determinism / plumbing-regression) with the
architecture confirmed line-by-line against source; v2 folds in the critique's
4 blockers + explicit rules. BUILD AGAINST THIS.
**Builder:** Fable. **Framework:** drops into Path B (merged `1ae6f86`).
**Brief:** `docs/s8a_path_a_eos_residency_brief_2026-07-20.md` · spec:
`docs/cuda_s8a_residency_spec_2026-07-19.md` (★ BUILD FINDING block).
**Gate:** `tests/cuda_s8a_check.py` extended (§7) — full-engine A/B tol 0 on a
space AND an ambient map, telemetry A/B, build-parity probe; per-kernel digest
gates untouched; NO re-baseline.

---

## 0. The load-bearing observation (critique-CONFIRMED)

`EOSSolver::step`'s only **global reductions** (order-sensitive scans) are in
the **pre-stage**, and every one consumes **tick-entry** state
(`eos_solver.cpp:260–392`): the `t_max_abs` max-scan (→ `c_local_q`), the
`max_rad` max-scan (→ `max_u`), the pre-Dalton `n_total` + `max_du` grad-scan
(→ `n_sub`). In the resident tick the **numpy mirror is authoritative at EOS
entry** (verified writer-by-writer: previous tick's D2H, tail, combustion,
FieldEdits, `stamp_units`, W3 seal, `destroy_wall` all write the mirror; the
resident water loop takes `atmosphere` const). So the whole host pre-stage
stays on the **host, verbatim, on the mirror** — bit-identical because it is
the same code on the same bytes. `n_sub` never needs a device→host sync.

Everything after the substep loop reads **post-substep** (device-only) state
and is per-cell: `div_u`, Dalton `n_total` (post), `pstar`, the whole MG
build (single-writer gathers + per-cell divides), vcycle/kick/compression,
the `atmosphere` store. **Path A needs zero deterministic parallel
reductions.** The two device reductions that exist (`boundary_flux` rail,
kick rail counters) are order-free integer atomics, digest-proven per-call.

## 1. Scope

**In:** the EOS stage of `_step_resident` becomes fully device-resident —
device MG build, device mid-stage, device-pointer vcycle/kick, zero mid-stage
plane transfers. The per-call `eos_step_cuda` path and the CPU path stay live
and byte-identical (shared bodies where factored). Residency stays opt-in
(`--resident`, default OFF).

**Out (unchanged):** combustion + tail brackets (S8c), traces (Path B's
`trace_smoke_resident`), CUDA graphs (S8b), `cast_fire_heat` (S8c). No
physics change. No golden re-baseline.

## 2. The resident tick after Path A (`PhysicsRunner._step_resident`)

```
1. host pre-physics (mirror)                          [unchanged]
2. water: from_host(5) → substeps resident → to_host(3) → host tail  [unchanged]
3. EOS pre-upload:  from_host(["atmosphere","wind_x","wind_y","temperature",
                     "gas","solid","is_vacuum","is_ambient",
                     "dyn_permeability"])
      ← replaces Path B's step-5 from_host + the per-call internal H2D.
      ★ BUILD FINDING (PART-1b catch): is_ambient is NOT static —
      destroy_wall's joins-ambient twin mutates it on a ring-adjacent breach
      (the ambient analogue of is_vacuum on space maps), so it rides the
      per-tick upload. This was ALSO a latent Path-B bug: the shipped trace
      loop reads device is_ambient as the trace sink and never re-uploaded
      it (the space-only Path-B gate could not see it) — fixed here for both
      consumers by this shared upload.
      NOTE: dyn_wave_absorb is NOT in this list — no device kernel in the EOS
      chain reads it (the kick consumes the host-hoisted absorb_q plane, §3.2
      step 4, computed FROM THE MIRROR — that is where body-shielding lives).
      The §5b from_host() DEFAULT set is untouched.
4. engine.run_substeps_resident(mirror arrays, dev ptrs, amb args)
      — host pre-stage on the mirror (verbatim §0, incl. p_prev:=atmosphere)
      — device: p_prev:=atmosphere (D2D) → cmask → substep loop →
        div_u/n_total/pstar → MG build → vcycle → kick/compression →
        atmosphere store.  ZERO plane transfers.
5. traces resident (unchanged call; its from_host is DROPPED — device
   gas/wind/perm/masks are already fresh from steps 3–4)
6. to_host(["atmosphere","wave_p","wind_x","wind_y","temperature","gas"])
      ← the once-per-tick synced-set D2H (locked Q4 decision); combustion +
        tail brackets read the mirror exactly as today.
      RULE: a defaulted to_host() is FORBIDDEN inside the resident tick — the
      device copies of heat/fire/wall_hp (and water fields on dormant ticks)
      are stale-by-design and would clobber the authoritative mirror.
7. combustion bracket (mirror)                        [unchanged]
8. tail bracket (mirror)                              [unchanged]
```

* `wave_p` is **not** uploaded in step 3: on device it is written (D2D) before
  any read (`debug_pstar_from_prev` reads it post-copy — matches the CPU).
  It IS in step 6's D2H (tail ripple reads it; bytes == the mirror copy the
  host pre-stage wrote, so the overwrite is identical — benign).
* `sponge_sigma` / `sponge_udamp` join `GameMap._RESIDENT_MASKS` (contiguous
  int32 `np.zeros`, reassigned only in `__init__` pre-residency — guard-safe).
  Like `is_ambient`/`floor_height` they are simply omitted from
  `_step_resident`'s explicit per-tick lists (static per map); a defaulted
  `from_host()` re-uploading them is harmless and allowed. `n_amb` stays a
  host `(n_gases,)` vector argument; `p_amb` a host scalar.
* Water-dormant ticks: step 3 uploads the full list regardless.
* **Mirror-currency invariant (comment in the runner):** every EOS input is
  current on the mirror at step-4 entry; Path A adds no host writer between
  step 3 and step 4.

## 3. C++ surface

### 3.1 `PhysicsEngine::run_substeps_resident` (physics_engine.cpp/.h)

Same numpy argument list as `run_substeps` (mirrors, used ONLY for the host
pre-stage + telemetry) **plus** raw device pointers (`uintptr_t` through
pybind, the Path-B idiom): `d_atmosphere, d_wave_p, d_wind_x, d_wind_y,
d_temperature, d_gas_base, d_solid, d_is_vacuum, d_dyn_permeability,
d_is_ambient(0), d_sponge_sigma(0), d_sponge_udamp(0)`.

Non-CUDA build: the method is declared unconditionally; its **body** is
`#ifdef BREACH_HAS_CUDA … #else throw std::runtime_error(...)` (the
`run_substeps` in-body idiom, physics_engine.cpp:287) so the binding compiles
and link-resolves on every build. With CUDA built it throws unless
`eos_step_backend_is_cuda()` (all four EOS kernel flags ON — no CPU fallback
for a device-pointer call). Calls `breach_cuda::eos_step_resident(this->eos,
...)`. No trace loop (Python drives `trace_smoke_resident` as today).

### 3.2 `breach_cuda::eos_step_resident` (new `cpp/src/cuda_eos_resident.cu`)

Added to `BREACH_SOURCES` → inherits the target-wide CUDA host `/fp:strict`
and `--fmad=false` (CMakeLists:101–107). Structure:

1. **Degenerate early-out** `n<=0 || dt<=0` — return, fields untouched.
2. **Host pre-stage on the mirror, verbatim** — factored into a shared
   `eos_host_prestage(...)` helper defined in `cuda_eos_step.cu` (ONE
   transcription, both entries call it). Its scope is `cuda_eos_step.cu`
   lines ~126–264 **inclusive of** (a) the `boundary_flux_` member reset —
   BOTH branches: ambient assign/zero AND the space-map `.clear()` (:131–137)
   — and (b) the **mirror `p_prev := atmosphere` copy** (:146). The copy is
   load-bearing twice: the `max_du` grad-scan (:210–212) reads `p_prev`, and
   it makes step-6's `wave_p` D2H land identical bytes on the mirror.
   Outputs: the per-tick scalar POD + `coeffE`/`coeffS` host planes + `cons`.
3. **Scalar-fold provenance (ONE transcription each — critique blocker):**
   - The four MG-build folds (`n_floor_q`, `gamma_q`, `dt_q`,
     `Kdt2dx2_raw` — the 5-op double expression with the divide,
     `eos_solver.cpp:741–752` incl. `std::max((double)dx,1e-6)`) are NOT
     re-derived in any CUDA TU. New exported helper
     `EOSSolver::mg_scalar_folds(float dt) const` in **eos_solver.cpp**
     (pure code motion of those lines; `mg_build_levels` itself calls it) —
     the proven MSVC `/fp:strict` TU stays the single source; the resident
     entry consumes the returned POD as kernel args.
   - The kick folds (`gamma_m1_q, t_min_q, t_max_phys_q, u_max_q,
     work_clamp_q, absorb_dt_q` + the shared `n_floor_q/dt_q/inv_2dx_q/
     Kdt_raw`) factor into a shared host helper in
     **cuda_kick_compression.cu** consumed by BOTH the per-call entry and
     the resident core — no second transcription in `cuda_eos_resident.cu`.
4. **Per-tick host-hoisted planes H2D:** `coeffE`, `coeffS`, `absorb_q`
   (the §2.5 hoist: `mul_q16(quantize(dyn_wave_absorb[i]), absorb_dt_q)`
   computed on the **mirror**). Per-tick inputs, not mid-tick traffic.
5. **Persistent scratch** (C++-owned, lazily (re)allocated — key
   `(h, w, n_levels)`, so a `use_multigrid`/`mg_min_dim` toggle re-keys):
   `d_svx,d_svy,d_st` (i32), `d_cmask` (u8), `d_coeffE,d_coeffS,d_dq_e,
   d_dq_s,d_scale` (i32), `d_absorb_q` (i32), `d_div_u,d_ntot,d_pstar` (i32),
   `d_rail` (u64×n_cons), `d_cnt` (u64×5), the **MG hierarchy** (≤9 levels ×
   {excl u8, m/gE/gS/recip/b/res i64, P i32}). Zero `cudaMalloc` steady-state.
   **PER-TICK ZERO RULE (critique blocker):** `cudaMemset(d_rail)` (ambient
   mode) and `cudaMemset(d_cnt)` run EVERY tick before the substep loop /
   kick — the per-call wrappers did this (`cuda_eos_step.cu:328`,
   `cuda_kick_compression.cu:336`); persistence must not skip it. In space
   mode the rail is not passed (nullptr per plane, `n_amb=0`) — matching the
   per-call gating (:326–329, :352).
6. **Device chain, zero plane transfers, ONE stream** (launch order = the
   CPU's pass order; the dependency argument below is stream-order):
   a. `p_prev := atmosphere` — `cudaMemcpyD2D`.
   b. `sl_cmask_build_device` (existing launcher).
   c. substep loop ×`n_sub`: D2D src snapshots → `sl_advect3_device` →
      `bulk_flux_plane_device` per cons plane (existing launchers, proven
      P6.5 chain, unchanged).
   d. `K_div_u` — per-cell, verbatim (`cuda_eos_step.cu:407–422`) incl. the
      BC ring `div_u=0` branch.
   e. `K_ntot` — per-cell Dalton over `n_gases` in **fixed gi order** (the
      CPU's per-cell add sequence). **Accumulate as
      `acc = (int32_t)((uint32_t)acc + (uint32_t)v)`** — defined wrap,
      byte-identical to the hosts' observed int32 wrap for all inputs
      (document in the kernel header; do NOT widen to int64-then-narrow —
      that would change bytes in the wrap case).
   f. `K_pstar` — per-cell, verbatim (:438–450); `debug_pstar_from_prev`
      passed as a flag (reads `d_wave_p` post-copy); the `< 0 → 0` floor
      applies on both branches.
   g. **MG build on device** — §4.
   h. `eos_mg_vcycle_resident` (§5) — identical schedule on the persistent
      hierarchy; `p_new` ≡ the post-`mg_zero_excl` `L0.P`, read in place
      (byte-equal to the per-call D2H'd `p_new`, solid cells zeroed — the
      ambient store's `solid ? p_new[i]` branch reads 0 both ways).
   i. `kick_compression_launch_resident` (§5) — `n_total` input = `d_ntot`
      (value-identical to the per-call recompute: same gas bytes, nothing
      writes gas in between).
   j. `K_store_atm` — `atmosphere := p_new` (+`p_amb` masked to `!solid` in
      ambient mode), verbatim (:527–533).
7. **Telemetry D2H (scalars only, documented):** `d_rail` → **assign** into
   `solver.boundary_flux_[cons[k]]` (the per-call semantics, :371–377);
   `d_cnt` → **`+=` accumulate** into the five cumulative members
   (:509–513). One `cudaDeviceSynchronize` at the end. A resident-call
   counter (`g_eos_resident_calls`) increments per call, bound as
   `eos_resident_calls()`.

### 3.3 Telemetry gaps (accepted, documented in the entry's header)

* **Digests** (`digest_advect`…`digest_compression`): NOT computed on the
  resident path (sequential host-side FNV; recomputing needs the D2H this
  patch removes). Grep-audited consumers: per-kernel gates + parity tests —
  all drive the per-call path (unchanged). Members left stale.
* **Debug probes** (`dbg_probe_idx`, `dbg_T_*`): unsupported on the resident
  path (probe debugging uses the per-call path).
* **Host `solver.levels_`** goes stale on the resident path (nothing calls
  the host `mg_build_levels`) — same gap class, same header note; debug
  tools reading `mg_levels()` after a resident run see old bytes.
* `dbg_last_c_local_q` / `dbg_last_n_sub` ARE set (host pre-stage — free),
  and the gate A/Bs them per tick (§7).

## 4. The MG build on device (the hard port — `mg_build_levels` → kernels)

Source of truth: `eos_solver.cpp:727–1012`. **The builder transcribes the
CODE, not this doc's prose** — every kernel body is a verbatim transcription
with `mul128_shr` → `mul128_shr_signed` (proven equivalent for all operand
signs at shifts 8/16/40) and `reciprocal_q16` → `reciprocal_q16_dev`
(instruction-identical). `n_levels` is a pure function of
`(h, w, mg_min_dim, use_multigrid)` — computed on the **host** (via
`mg_scalar_folds`' sibling or inline, same code). The level loop is a host
loop over launches: each level's build reads the previous level's finished
arrays (launch boundary = the CPU's level-loop boundary).

**THE EVERY-CELL-WRITE RULE (load-bearing).** The CPU `assign(n,0)`s exactly
8 arrays per level (`excl,m,gE,gS,recip,P,b,res` — :781–788, :902–909), then
writes subsets. The persistent hierarchy carries last tick's bytes, so every
build kernel writes **every cell of every output array unconditionally** —
else-branches write the 0 the `assign` left. (`res` needs no build write:
the vcycle's residual pass writes every cell before restrict reads it, and
the flat path never touches it — critique-verified.)

| array | writer kernel | notes |
|---|---|---|
| L0 `excl` | K_L0_excl | branch priority verbatim: `solid→2` else `is_vacuum→1` else `(ambient && is_ambient)→1` else 0 (:789–798) |
| L0 `m,b,P` | K_L0_mbP | every cell; `excl!=0 ⇒ m=b=P=0`. Verbatim sequence (:805–834): `gp_raw = mul128_shr(γ_q, pstar[i],16)`; `aK = mul128_shr(gp_raw, Kdt2dx2_raw,16)`; **`if (aK<1) aK=1;` THEN `m = (((int64_t)1)<<32)/aK;` THEN `if (m<1) m=1; if (m>M_CAP) m=M_CAP;`** (M_CAP = `((int64_t)1)<<38`; the int64-typed shift literals are mandatory — a bare `1<<32` is UB); `gp_dt = mul128_shr(gp_raw, dt_q,16)`; `rhs = pstar[i] − mul128_shr(gp_dt, div_u[i],16)`; ambient: **`rhs -= p_amb` BEFORE the m-multiply**; `b = mul128_shr(m, rhs, 8)`; warm start `P = (int32_t)((int64_t)p_prev[i] − (int64_t)p_amb)` in ambient (widen-then-narrow, wrap-deterministic) / `p_prev[i]` in space |
| L0 `gE,gS` | K_L0_faces | every cell (0 at guard-fail and at the x=w−1 / y=h−1 boundary). **The guard is `excl!=2` on BOTH endpoints, NOT `==0`** (:845, :856) — regular↔Dirichlet faces carry conductance; the Galerkin anchor depends on them. Body: `pf = min(perm[i],perm[j])` (`std::min` select semantics `(b<a)?b:a` — not fminf), `pf>0` gate, `nhat = (q16)(((int64_t)n[i]+n[j])>>1)`, **`if (nhat<n_floor_q) nhat=n_floor_q`**, `g = mul_q16(quantize((double)pf), reciprocal_q16_dev(nhat))` |
| L0 `m` σ | K_L0_sigma | launched only when `ambient_mode && d_sponge_sigma`; per-cell `excl==0 && s>0`: `ms = m + s; if (ms > M_CAP) ms = M_CAP; m = ms` (:877–886 — the **re-clamp is reachable**, do not drop). Runs AFTER K_L0_mbP (b uses un-σ'd m) and BEFORE coarse + recip (both fold σ) — the CPU's exact placement |
| coarse `excl,m,b,P` | K_C_excl_m | every cell: excl rule (all-vac→1, all-nonregular→2, any-regular→0), `m=0` unless regular then `min(m_sum + anchor, M_CAP_L)` (`M_CAP_L = ((int64_t)1)<<44`); child loop in the CPU's `dy,dxx` order with the `fy>=F.h/fx>=F.w` clipping; the **anchor** loop verbatim (:940–953, reads fine `gE/gS/excl` incl. the out-of-block `gE[fi−1]`/`gS[fi−F.w]` reads — read-only fine data); writes `b=0, P=0` (mirrors assign; vcycle rewrites both before read — belt-and-braces) |
| coarse `gE,gS` | K_C_faces | every cell; crossing-fine-face sums with the **both-`==0`** guard (different from L0! :970, :982) |
| `recip` (all levels) | K_recip | every cell; `excl!=0 ⇒ 0`; diagonal fold in the CPU's E,W,S,N order with the `excl!=2` neighbor guards; `if (d_raw<1) d_raw=1`; `recip = (((int64_t)1)<<48)/d_raw` |

**Numeric-exactness arguments (corrected per critique):**

* **Integer divides on device** (`m`, `recip`): both strictly
  positive/positive (`gp_raw ≥ 0` since `γ_q>0`, `pstar ≥ 0` floored on both
  branches; `d_raw ≥ m ≥ 1`); C++ truncation-toward-zero is identically
  specified on MSVC and nvcc; nvcc's int64 divide is exact software emulation.
* **`quantize((double)pf)` on device** — the first float arithmetic admitted
  into device sim code (supersedes the P6.3 "no float on device" scope note,
  which was a per-patch scope choice, not a determinism requirement). The
  claim is NOT "every step is exact" (false for denormal `pf`: `scaled+0.5`
  can round). The claim is: `(double)pf` exact; `pf*65536.0` exact (power-of-
  two exponent shift); `scaled+0.5` is **one identically-rounded IEEE-754
  double add** (MSVC `/fp:strict` SSE2 ≡ device); the truncating cast is
  identical. FMA contraction is harmless precisely because the product is
  exact (`fma(v,65536,0.5) = RN(exact+0.5)` = the plain path). Pins:
  arithmetic in **double** end-to-end (a `+0.5f` re-rounds at 24 bits), no
  `--use_fast_math` on the TU (CMake already sets `--fmad=false`), `std::min`
  select semantics for `pf`.
* **`mul128_shr` ≡ `mul128_shr_signed`**, **`reciprocal_q16` ≡
  `reciprocal_q16_dev`**: verified identical (cuda_fixedpoint_device.cuh:40–46,
  :112–137), already digest-proven at the used shifts.

No atomics, no cross-thread traffic; every output cell has exactly one
writer; every read is of a buffer completed by a previous same-stream launch
(dependency-completeness critique-verified kernel-by-kernel, incl. the
lv≥2 case reading the previous host-iteration's K_C_faces output).

## 5. Reworks of the proven kernels (plumbing, no math)

* **`cuda_mg_solve.cu`** — factor the schedule body (launch loops + fused
  tail + `mg_zero_excl` + the `launches_actual/naive` counting) into a
  file-local `run_schedule(MGLevelDev dev[], ...)`. New exported
  `eos_mg_vcycle_resident(...)` takes the persistent hierarchy as a
  plain-pointer `MGLevelDevPtrs` header struct (ints + raw pointers only —
  the `MGLevelHostView` precedent; ABI-clean, `MGTailArgs` stays
  file-local), runs the identical schedule, NO upload/D2H/digest. The
  per-call `eos_mg_vcycle` keeps its upload/digest wrapper around the shared
  body — `cuda_mg_solve_check` (bytes + launch-count assert) pins any drift.
* **`cuda_kick_compression.cu`** — factor
  `kick_compression_launch_resident(...)` (device pointers, pre-folded
  scalars, launches K1+K2 only — no malloc/transfer/**memset**/sync/digest;
  the caller owns `d_cnt` zeroing per §3.2.5) + the shared scalar-fold
  helper (§3.2.3). The per-call wrapper keeps its existing
  H2D → memset → core → D2H → digest flow. Declared in `cuda_resident.h`.
* **`cuda_eos_step.cu`** — the pre-stage factors into `eos_host_prestage`
  (§3.2.2, pure code motion; per-call behavior byte-identical).
* **`eos_solver.cpp/.h`** — `mg_scalar_folds` extraction (§3.2.3, pure code
  motion; `mg_build_levels` calls it — CPU bytes unchanged, pinned by the
  existing CPU goldens in `cuda_eos_step_check` PART 2).

## 6. Python + bindings

* `bindings.cpp`: bind `run_substeps_resident` on the engine unconditionally
  (the method body throws on non-CUDA builds — §3.1; the free-function
  resident bindings stay inside the `#ifdef` as today). Bind
  `eos_resident_calls()`. `0 → nullptr` for the ambient statics.
* `gamemap.py`: `sponge_sigma`/`sponge_udamp` → `_RESIDENT_MASKS` (§2 note).
* `physics_runner.py`: rework `_step_resident` steps 3–6 per §2.

## 7. The gate (extend `tests/cuda_s8a_check.py`)

* **PART 1a (space map)** — the existing 40-tick full-engine A/B, tol 0, all
  synced fields, scripted breach — now exercises the resident EOS. Added
  per-tick **telemetry A/B** (all already Python-bound): `dbg_last_n_sub`,
  `dbg_last_c_local_q`, the five `*_hits` counters, `eos.boundary_flux()`.
  Vacuousness guards: `bp.eos_resident_calls()` strictly increasing across
  the GPU run AND `bp.eos_step_cuda_calls()` NOT advancing (proves the
  bracket is actually gone, not silently falling back).
* **PART 1b (ambient map — critique blocker):** a second A/B leg on an
  ambient-ring scenario with nonzero `sponge_sigma` + `sponge_udamp` and a
  ring-adjacent breach — covers the resident device ambient branches
  (shift/re-shift, ring excl, σ-fold, ring div_u=0, masked `+p_amb` store,
  rail accumulation, udamp kick path). Same tol-0 field + telemetry compare
  (boundary_flux per tick).
* **PART 1c (build-parity probe — critique blocker):** a test-only binding
  (`eos_mg_build_parity(...)` or debug D2H) that, on identical inputs, runs
  host `mg_build_levels` and the device build and byte-compares every
  level's `excl/m/gE/gS/recip/b/P` — localized proof for the hardest port,
  over the PART 1a/1b scenarios' states (crafted edge coverage: odd dims →
  child clipping, all-vacuum/all-solid coarse cells, M_CAP rows). Test-only;
  not on the hot path.
* **PART 2** — add the isolated **EOS-stage** tax bench: per-call
  `run_substeps(do_traces=False)` (backends ON) vs
  `from_host(8) + run_substeps_resident + to_host(6)` at ≥2 grid sizes;
  resident must clearly win, and win MORE on the bigger grid. Bench comment
  notes the per-call side also pays ~6 host FNV plane digests the resident
  side skips — part of the margin is digest removal, not transfer removal
  (do not over-claim). Existing water/smoke assertions stay.
* Per-kernel digest gates + full `pytest tests -q`: green, untouched.

## 8. Build order (each piece compiling + gated before the next)

1. Pure-code-motion factorings (`eos_host_prestage`, `mg_scalar_folds`, kick
   folds + launch core, vcycle schedule body) — per-kernel gates + CPU
   goldens prove zero drift BEFORE any new code runs.
2. `cuda_eos_resident.cu`: scratch + substep chain + mid-stage kernels.
3. The MG build kernels + the PART 1c parity probe (gate the build in
   isolation FIRST).
4. Orchestration + engine method + bindings + runner rework.
5. Gate: PART 1a/1b/1c + PART 2 + full suite.

## 9. Accepted-risk register

Stale digests/probes/levels_ on the resident path (§3.3, header-documented);
the ~56 B/tick telemetry D2H (rail + counters — live consumers keep working);
`PART 2`'s margin partly digest-skip (documented in the bench).
