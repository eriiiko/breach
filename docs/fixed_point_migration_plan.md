# Breach Fixed-Point (Q16.16) Migration Plan

**Status: PLAN — prepared by an autonomous research + 6-reviewer panel run (2026-06-23). Awaiting Erik's review. NOT yet implemented.**

This is the third sibling of `docs/physics_engine_unification_plan.md` (Patch 1, shipped, 0-ULP) and
`docs/patch2_dt_policy_plan.md` (Patch 2, shipped, feel-gated). It is the **fixed-point arithmetic
patch**: convert the entire sim-state path from float32 to Q16.16 integer, **tested on CPU**, as the real
prerequisite for the CUDA port and for cross-GPU lockstep multiplayer.

It builds on **locked decisions** (do not relitigate): full fixed-point, cross-GPU determinism (must
survive *different* GPUs), Q16.16 = one int32, Nvidia-only acceptable, fixed-point-before-CUDA, render
buffers stay float. `temperature_solver.{h,cpp}` is the working production template.

> **How this plan changed after the panel.** Six adversarial reviewers (numerical-methods,
> determinism-adversary, gpu-perf-skeptic, lockstep-practitioner, scope-risk, research-grounding) read the
> draft against HEAD. They converged on **eight blocker-class corrections** that the draft got wrong or
> missed. The biggest reversals from the draft: (1) **drop Chebyshev-Jacobi** — it re-imports the very
> non-associativity it was meant to kill (its weights need a runtime spectral estimate); keep
> **fixed-schedule Red-Black GS** as canonical. (2) **The float substep cliffs (`n`, `n_smoke`, water `n`)
> are an un-migrated cross-GPU desync the draft left in float** — they become a first-class S0/S1
> deliverable. (3) **The temperature "difference-then-shift" idiom is NOT conservative** — copying it into
> the conserved fields (atmosphere/smoke/gas/water) installs a deterministic-but-wrong mass leak; conserved
> fields need an **edge-flux** idiom instead. (4) **The determinism contract leaks past C++ into Python**
> (`combat.py` does float damage/kill math on the integer `heat` field). (5) **The GS per-cell divide must
> be a precomputed-reciprocal multiply from day one** — a 64-bit integer divide in the GS inner loop is the
> worst op on Ampere. (6) **De-risk the GPU premise with a one-week spike before the multi-week migration.**

---

## 1. The research-grounded WHY

### 1.1 The goal restated as a falsifiable contract

We want **bit-identical sim state across different GPUs** — the same int32 field bytes on the RTX 3070, on
a future Blackwell card, and on the CPU reference build. That single falsifiable claim is the *only* thing
that makes (a) lockstep multiplayer and (b) the CUDA port trustworthy. Fixed-point is **not** "more
accurate": it trades float's rounding/range problems for overflow/precision-budget problems (ACCU Overload
100, *"Why Fixed Point Won't Cure Your Floating Point Blues"*). The one thing it buys that float cannot
give us cross-GPU is **bit-reproducibility** — which is exactly the thing we need.

The panel sharpened the contract into **two separately-testable properties** (do not conflate them):

- **P1 — cross-order / cross-arch bit-identity.** The same logical computation gives the same int32 bytes
  regardless of reduction order, SIMD width, warp size, or GPU architecture.
- **P2 — physical conservation to the LSB.** Conserved quantities (total atmosphere in a sealed region,
  total smoke, total water) do not drift over a long settle.

The draft assumed integer associativity buys P2 for free. It does not. A solver can be **perfectly P1-true
and still deterministically leak mass on every peer** — staying green forever while the game slowly breaks
(rooms lose air, floods grow). Both properties get a CPU-only test in S0 (§6), *before* CUDA exists, which
is where the locked sequence says determinism must be proven.

### 1.2 Why float is not bit-deterministic cross-GPU — the failure modes that actually bite us

IEEE-754 mandates `+ − × ÷ sqrt` are **correctly rounded** on every conforming target (x86, ARM, every
CUDA GPU). So the divide is *not* the villain (NVIDIA, *Floating Point and IEEE 754*; consistent with
`docs/gs_precomputed_reciprocal_techniques.md`). What IEEE-754 does **not** pin down is everything *around*
the ops — and those are the real desyncs, ranked by how hard they bite *our* code:

| Rank | Failure mode | Where in Breach | Fixed-point status |
|---|---|---|---|
| **1** | **Non-associative parallel reduction** | `mean_wp` global sum `atmosphere_solver.cpp:143-151` (`sum += wave_p[i]` → `sum/count`). On GPU this becomes a reduction tree whose result depends on warp size / block count / scheduler → **differs across architectures**. The deepest latent desync. | **Solved for free.** Two's-complement integer add is associative + exact → any order gives the identical sum. The single strongest argument for the migration. |
| **2** | **Control-flow cliffs over a float reduction** (NEW — panel) | `n = ceil(sim_time/dt)`, `n_smoke = ceil(...)` over the `max_wind_sq` float reduction, water `n` (`physics_engine.cpp:124-128,182-198,287`). A 1-ULP slip flips an **integer substep count** → two peers run a *different number of iterations* → total desync; fields are bit-identical *within* a substep so a naive digest misleads. | **NOT auto-solved.** Today these are deliberately float64-matched-to-numpy (a *same-binary* guarantee, useless cross-GPU). Must become fixed-point too (§4.4). |
| **3** | **FMA contraction** | `rhs[i] + mu*nb`, `1+mu*wsum` (`:257`); IMEX transfer `(wave_p[i]-mean_wp)*xfer` (`:157`); wave kick `c_sq*lap` (`:108`). `ptxas` fuses by default; MSVC contracts under `/fp:precise` too. | Eliminated in the integer path; but **must be killed at every remaining float site** (load-time bakes, render) so it cannot leak (§6.1, M1). |
| **4** | **fast-math reassociation + reciprocal substitution** | The 4-term `nb`/`wsum` sums (`:251-255`); the divide may become `RCPSS`-class. **We ship `/fp:fast` today (`CMakeLists.txt:14,16`).** | Removed by construction in integer; audited away at the residual float sites. |
| **5** | **Red-black GS read-after-write order** | The in-place sweep (`:199-261`). | **Orthogonal to Q-format.** *But it is NOT a cross-arch hazard* — a fixed two-color schedule fully determinizes it (red reads only black, black reads only red; no intra-color RAW). Addressed by pinning the schedule (§3), not by Jacobi. |
| **6** | **Transcendentals (`exp/sin/cos/pow/sqrt`)** | Sim-path `sqrt` (fire wind-mag, smoke back-trace, water CFL), ray `sin/cos`; render-only `exp` exempt. Device `sqrtf` is 0-ULP but `sinf/cosf`=2 ULP, `expf`=2 ULP, `powf`=4 ULP, "may differ across device architectures." | Must become deterministic integer routines / committed LUTs (§5). |
| **7** | **x87/SSE intermediate precision** | CPU golden-reference build only. | Irrelevant once integer. |

### 1.3 The integer guarantee and its real caveats

Two's-complement integer `+ − ×` and shifts are **exact and associative** — no rounding step, no
order-dependence, no architecture-dependent intermediate precision. PTX defines integer behavior as "fully
defined in all cases" (`shr.s32` arithmetic, `mul.wide.s32` exact 64-bit), bit-exact on **any** CUDA GPU.
This is exactly what `temperature_solver.{h,cpp}` exploits in production. The caveats we must *design for*:

1. **Multiply needs a 64-bit intermediate.** `Q16.16 × Q16.16` has a 64-bit exact product; `(int32*int32)>>16`
   in 32 bits overflows (signed-overflow UB). Always `(int64_t)a*b >> 16` (temperature does this,
   `temperature_solver.cpp:83`). **GPU caveat (§8):** int64 multiply is *emulated* on Ampere — use the
   narrowest provably-safe intermediate.
2. **Signed overflow is UB.** Need saturating arithmetic (`heat_saturating_add`, `raycaster.h:37`) or
   provably-bounded ranges per field, with a **declared overflow policy** per field (§2.4). The main new
   design burden.
3. **Division is the sharp edge.** Integer `/` is bit-exact cross-arch but **truncates toward zero**;
   `Q16.16/Q16.16` needs the numerator pre-shifted into 64-bit first. The GS dynamic divide must become a
   **precomputed-reciprocal multiply** (§3, §8) — a per-cell int64 divide is unshippable on GPU.
4. **Signed-shift portability.** Right-shift of a negative was implementation-defined pre-C++20; left-shift
   of a negative is UB. **Pin `-std=c++20` host + device in CMake with a `static_assert`** (panel,
   research-grounding) — the whole shift/division determinism argument silently depends on it. Then choose
   the shift idiom *per use* (§2.3, R2′).

### 1.4 What the literature and shipped practice endorse for *our* case

The split across 25 years of post-mortems is clean: **same-binary lockstep → constrained float is enough**
(Age of Empires, `/fp:strict`, no fast-math); **cross-vendor / cross-GPU lockstep → fixed-point**. The
*current* reference implementations of Erik's exact architecture (fixed-point sim + float render) are
**Photon Quantum** and the **FixPointCS** library (purpose-built for bit-exact cross-platform
sqrt/sin/cos/exp) — a more relevant oracle than `fpm`/`libfixmath`, which do **not** advertise
cross-platform bit-exactness. The universal rule from the 2024-2025 lockstep guidance (yal.cc, SnapNet):
*no third-party float library and no host `Math.Sqrt` in the deterministic path; render stays float, sim
becomes integer; checksum every tick; keep a forensic dump.*

For the CUDA port specifically, NVIDIA's CCCL determinism tiers make it explicit — cross-GPU **float**
reductions need the special Reproducible Floating-point Accumulator (RFA) at 20-30% cost; **integer**
reductions need none of it (they run at the relaxed-tier two-pass cost — deterministic, not literally
"free"). **Fixed-point on GPU is not a workaround — it is the path that makes our hardest failure mode
(rank #1) disappear.**

---

## 2. The Q16.16 format decision + per-field table

### 2.1 Decision

**Adopt Q16.16 (one int32) as the uniform default for every sim-state field**, with mandatory non-uniform
exceptions and a fixed house style. `1.0 == HEAT_SCALE == 65536` raw counts (same constant as
`heat`/`temperature`, `raycaster.h:25`). Resolution δ = 2⁻¹⁶ ≈ 1.526e-5; representable magnitude ≈
±32768. One scale for *storage* so cross-field reads never need a rescale.

Why Q16.16 over the alternatives (defensible against the panel):

- **Q24.8** buys range we don't need and pays 256× coarser resolution (δ≈3.9e-3) — catastrophic for a 0..1
  diffusion field where late-time per-step increments are routinely sub-LSB.
- **Q8.8 / 16-bit** is a *later, profiling-gated* bandwidth optimization (locked decision defers it). See
  §2.6 for the panel's caveat: **freeze the per-field GPU widths now** even though storage stays int32 on
  CPU, so the CUDA buffers do not need a second cutover.
- **Uniform int64** (the BOID approach) doubles bandwidth — the opposite of the CUDA memory-bound goal.
  Reserve int64 for **accumulators and intermediates**, never for storage.
- **Posits** have no NVIDIA hardware → software-emulated, slow, and the emulation must itself be
  bit-identical (reintroduces the problem). Rejected.

### 2.2 The mandatory non-uniform exceptions

1. **`mean_wp` accumulator → int64**, no saturation (a deterministic int64 sum is the whole point; it
   cannot realistically reach 2⁶³ — §4.2). Mirrors AMBER-SPFP's wide accumulators for reproducible force
   sums.
2. **All multiply / flux intermediates → int64, narrow once at the end** (`temperature_solver.cpp:71,83-85`
   is the template). The wave-velocity integration `wave_v += (c_sq*lap − …)*dt` is the headline overflow
   watch-point (§2.4, M5).
3. **`wave_v` is the most likely per-field-format exception.** Decide by measurement (§2.4): a blast can
   drive it toward the ±32768 ceiling. If the measured peak is thin on headroom, `wave_v` (and possibly
   `wave_p`) go Q24.8 *for those fields only*, ahead of the gas planes.

### 2.3 The house style (the rules, refined by the panel)

- **R1 — int64 intermediate on every multiply**, narrow once: `mul(a,b) = (int32)(((int64)a*b + round) >> 16)`.
  Round mode is **per-operation** (R2′ below), not one global rule.
- **R1′ (GPU) — narrowest provably-safe intermediate.** int64 multiply is emulated on Ampere (§8). Where a
  product's magnitude is provably bounded in 32 bits (most [0,1]-field stencil terms), keep it 32-bit. Where
  int64 is genuinely required (`c_sq*lap`, the `mean_wp` accumulator, the GS numerator), keep it. The
  per-field table (§2.4) carries the proof.
- **R2 — shift the DIFFERENCE for relaxation fields** (`(field[n]-field[i])` scaled, accumulated, narrowed
  once). Equal neighbours give *exactly* 0 → drift-free rest state. **This is correct ONLY for
  non-conserved relaxation fields (temperature, heat).** It is the temperature template verbatim.
- **R2-CONS — edge-flux for CONSERVED fields** (atmosphere, smoke, gas, water_depth). **(BLOCKER fix — the
  draft's single R2 was wrong here.)** The temperature gather shifts each cell's difference *independently*;
  for an odd difference the i→n flux and the n→i flux are not negatives of each other under arithmetic
  right-shift → a 1-LSB-per-asymmetric-face mass leak every tick. Temperature gets away with it because heat
  is *not* conserved (cooling sheds it by design). Atmosphere/smoke/gas/water **are** conserved (that is the
  entire reason `mean_wp` exists). Compute each face flux **once** and apply it with opposite sign to both
  cells: `flux = mul(diff, w_face); field[i] += flux; field[n] -= flux;`. Round-to-nearest-even on the flux,
  or carry the truncation remainder (§8 carry-save idea), if exact rest is needed. **This is the same
  refactor that retires `mean_wp` (§4.5) — recognize it as the natural endpoint, not a footnote.**
- **R2′ — pin the shift/divide rounding mode per use:**
  - *flux-difference shifts* (R2/R2-CONS): plain arithmetic shift toward −∞ (matches temperature; the i+n
    pair sums conservatively).
  - *standalone signed-magnitude decay* (e.g. cooling): sign-symmetric toward-0 `x<0 ? -((-x)>>s) : x>>s`
    (`temperature_solver.cpp:142`) — gives the dead-band rest state.
  - *reductions that must match PTX `div`* (`mean_wp` `sum/count`): true integer `/`, truncating toward 0.
  - The draft implied one rule; there are three. Which applies is decided by whether the value is a
    conservative flux pair, a magnitude decay, or a quotient. **State it at every site.**
- **R3 — saturating add for unbounded accumulators** (`heat_saturating_add`); structural no-ops as
  `continue`/sentinel skips, never `*0.0`/`+0.0` (fast-math folds those inconsistently — the
  `water_solver.cpp:31-44` `zeros()` scratch exists for exactly this reason).
- **R4 — signed-multiply rounding is NOT `+0.5`.** `heat_quantize` (`raycaster.h:28-34`) rounds half-up,
  which is asymmetric for negatives (−2.5 → −2). For signed fields (`wave_p`, `temperature`) use
  round-half-away-from-zero or round-half-to-even, defined once, normatively.

### 2.4 Per-field RANGE / PRECISION / OVERFLOW / SMALLEST-INCREMENT table

At shipped config (`config.toml`): `wave_c=66 → c_sq=4356`, `d_atm=200`, `ticks_per_second=24 → dt≈0.0417`,
`mu = d_atm·dt ≈ 8.33`, `transfer=0.5 → xfer = 0.5·dt ≈ 0.021`, `damping=3.0`, `ratio_cap=1.5`,
`ceiling_h=2.5`, `v_max=8.0`.

| Field | Range | Signed | Conserved? | Overflow policy (cite bound) | Smallest meaningful increment | GPU width (frozen) |
|---|---|---|---|---|---|---|
| **atmosphere** | ~1.0 interior; →0 drained; plume up to ~2.0 | no | **yes → R2-CONS** | **Provably-bounded.** GS numerator chain `(rhs<<16)+mul(mu,nb)` then `<<16`: `bits ≤ 33+16 = 49`, ~14 bits int64 headroom. `static_assert` on `max(mu)` (grows if `d_atm`↑ or tps↓). | transfer `xfer·(wave_p−mean_wp)` at small late anomaly ~1e-3 → **~1.3 LSB** — round-half here, watch | int32 |
| **wave_p** | zero-mean anomaly; source 8-10 | **yes** | (mass-neutral via mean) | MED. Kick chain in int64, **apply dt before narrowing** (M5). | source-scale, comfortable | int32 (Q24.8 candidate, §2.2) |
| **wave_v** | tens-hundreds in a blast | **yes** | no | **MED→HIGH watch-point.** `c_sq·lap` ≈ 4356·40 ≈ 1.7e5 in real units — **exceeds Q16.16 ±32768 BEFORE `·dt`**. Safe ONLY if the int64 intermediate carries `c_sq·lap` and `·dt = /24` is applied *before* narrowing. Harness must measure peak `\|wave_v\|`. **Most likely Q24.8 exception.** | blast-scale | int32 or Q24.8 (measure) |
| **wind_x / wind_y** | gradient O(0.1-1), shockwave spikes | **yes** | no | LOW. 2-term difference × 0.5 (`>>1`). Feeds smoke advection + `max_wind_sq` cliff → must share format. | small late-time gradient `>>1` — round-half | int32 |
| **smoke (BLACK_SMOKE)** | **[0,1] clamped** (`smoke_dynamics.cpp:213`) | no | **yes → R2-CONS** | LOW. Bilinear renorm divide (M8) is a *second* dynamic divide — int64 `(acc<<16)/wsum`, small-divisor guard, or fixed-weight LUT. | density increments comfortable | **int16 (Q1.15)** |
| **each gas plane (×5)** | **[0,1] clamped** | no | **yes → R2-CONS** | LOW. 5 planes = widest sim memory. | comfortable | **int16 (Q1.15)** |
| **fire** | **[0,1] clamped** (`fire_simulation.cpp:144`) | no | no (logistic) | **MED precision (M2).** 6-factor `grow` product (`:92-95`): chained `mul64` truncates each step → do the whole chain in **Q16.48 int64, narrow once** at `I + dt·(grow−die)`; **pin the multiply tree order**; `smoothstep` cubic + `clamp01` need pinned fixed-point definitions. Output flips burning↔extinguishing → discrete desync vector. | logistic edge is sensitive | **int16 (Q1.15)** |
| **water_depth** | metres, 0..~2.5 | no | **yes → R2-CONS** | LOW (δ=15µm). Cleanest range. | comfortable | int32 |
| **flow_vx / flow_vy** | m/s, clamped ±8.0 | **yes** | no | LOW. Sign-symmetric shift. `dt/dx`, `dsdx` → precompute/scalar (§5). | comfortable | int32 |
| **heat** | Q16.16 already, saturating | no | no | shipped — `heat_saturating_add` | — | int32 |
| **temperature** | Q16.16 already (ΔT) | **yes** | no | shipped — sign-symmetric cooling | — | int32 |
| **permeability** | `{0, 0.5, 1.0}` via `min(perm_i,perm_n)` per face | no | — | quantize Q16.16; fold `w∈{0,½,1}` → shift (M7) | exact | int32 / uint8 bucket |
| **render-only:** light_rgb, smoke_glow, ripple, ripple_v | — | — | — | **STAY FLOAT, out of scope** | — | float |

### 2.5 Idiom summary

- **Multiply:** `(int64)a*b >> 16` + per-op rounding; R1′ narrowest-safe intermediate on GPU.
- **Conserved diffusion:** edge-flux (R2-CONS), once per face, opposite signs.
- **Relaxation diffusion:** difference-shift (R2), gather, narrow once.
- **Divide (rare — three sites, see §3/§4/M8):** numerator pre-shifted into int64; truncate toward 0;
  guard divide-by-zero; **forbid the compiler from magic-number-reciprocal'ing it** (§6.1).
- **Divide-by-constant:** load-time precompute → runtime shift/multiply (the temperature
  `heat_inv_shift`/`face_shift` pattern; generalize to *every* divide-by-config in fire/smoke/water — M7,
  m7).

### 2.6 Freeze the GPU field widths now (panel: gpu-perf-skeptic)

Q16.16/int32 buys **zero bandwidth** over the float32 it replaces — same byte footprint. A diffusion
stencil on a 448 GB/s 3070 *will* be bandwidth-bound, so 16-bit for the [0,1]-clamped fields is not "if
profiling shows it" but "yes, for smoke + 5 gas planes + fire." Ship int32 on CPU for simplicity if you
like, but **record the per-field GPU width in the Q-format version tag (§6.2.7) from the start** so the
digest schema and CUDA buffers are designed once. int16 add/mul are still associative/exact — zero
determinism cost. (Later: packed int16 pairs exploit Ampere's `vadd2`/`__dp2a` SIMD-within-register paths.)

---

## 3. The Gauss-Seidel divide + GS-vs-Jacobi verdict

### 3.1 The divide in fixed-point — precomputed reciprocal, NOT a per-cell integer divide

`atmosphere[i] = (rhs[i] + mu*nb) / (1 + mu*wsum)` (`atmosphere_solver.cpp:257`) is **one** of (at least)
**three** genuine dynamic-divisor divides in the migration — the draft wrongly called it "the one." The
others: the smoke bilinear renormalization `acc/wsum` (`smoke_dynamics.cpp:122`, M8) and `mean_wp`
`sum/count` (§4).

The draft proposed a per-cell `(numer<<16)/denom` int64 divide in the GS inner loop. **The panel
(gpu-perf-skeptic) flags this as a blocker:** NVIDIA GPUs have *no* integer-divide instruction; 32-bit
`int/int` lowers to ~15-20 instructions, and **64-bit divide has no fast path**. Running it ~16×/cell/tick
(`gs_iters=8` × 2 colors, diffuse now runs once/tick — `physics_engine.cpp:164`, **not** the stale 48)
would dominate the CUDA runtime for *zero* determinism benefit (the reciprocal-multiply is equally
bit-exact).

**So the precomputed reciprocal is the baseline, not a deferred optimization.** Per tick, precompute a
Q16.16 inverse-diagonal `Dinv[i]` once; the sweep is a multiply:

```
// once per tick (rebuild trigger keyed on mu | obstacles | is_wall | is_vacuum | permeability):
Dinv[i] = reciprocal_q16(ONE_Q16 + mul(mu, wsum_i));   // explicit `continue` on skip, NOT Dinv=0
// hot sweep, in residual/flux form (NOT the quotient form — see below):
atm[i] += mul( Σ_face mul(mul(mu, w_face), (atm[n]-atm[i])) - (atm[i]-rhs[i]) , Dinv[i] );
```

This collapses N×16×divide into N×reciprocal(once) + N×16×multiply. The reciprocal itself is the one slow
divide, done once per cell per tick (or only on changed cells with delta-tracked `stamp_units`, per the gs
doc).

### 3.2 Use the FLUX/RESIDUAL form, not the quotient form (BLOCKER — numerical-methods)

The draft wrote the **quotient** form `atm = (rhs + mu·nb)·Dinv` in §3.1/§4.2 but the **residual** form in
§3.3 — a contradiction. The quotient form **does not have a fixed point at the analytic solution under a
truncating multiply**: with `mu·wsum ≈ 33` the operator has gain, so the sub-LSB truncation deficit is
re-injected and amplified every sweep → a systematic mass leak (not a symmetric dither). The temperature
safety analogy (convex combination → provably in-range) **does not transfer**: temperature enforces
`SHIFT_MIN==2 → Σr ≤ 1` (a convex combination, `temperature_solver.h:34-37`); the atmosphere operator has
no such max-principle guard. **Use the residual/flux increment form** (above): equal neighbours at the
fixed point → increment truncates to exactly 0 → drift-free, and the int32 narrowing is provably bounded by
the per-sweep flux bound `≤ mu·wsum·(max−min)·Dinv`. Add a debug `assert` on the post-sweep range (no
free theorem hands you the bound).

### 3.3 GS vs Jacobi — **keep fixed-schedule Red-Black GS; DROP Chebyshev-Jacobi**

This is the plan's biggest reversal from the draft. The draft recommended Chebyshev-accelerated Jacobi as
canonical. **Four reviewers independently rejected it.** The reasons:

1. **Chebyshev re-imports the non-associativity we just killed (determinism-adversary, numerical-methods).**
   Its acceleration weights ω_k are non-stationary per-iteration scalars derived from the operator's
   spectral bounds [λ_min, λ_max], which depend on `mu` (per-tick, since `dt` floats) **and** on
   permeability/breach geometry (which changes mid-trajectory — the harness even forces `destroy_wall(8,0)`).
   So you must either (a) **re-estimate eigenvalues at runtime** — a power iteration with a *global
   reduction* and a fixed-point convergence test, i.e. re-creating the hardest determinism problem inside
   the smoother; or (b) freeze stale ω_k → Chebyshev is no longer optimal and can **diverge**.
2. **Chebyshev is a second-order recurrence, provably *more* perturbation-sensitive than first-order
   Jacobi.** In fixed-point, perturbation = quantization noise on every weighted SAXPY; the
   error-amplification near the spectrum endpoints can *lift* the quantization floor. You may converge
   faster in float and worse in Q16.16.
3. **Plain Jacobi at our `mu` does not converge in 8 sweeps (numerical-methods).** `ρ_Jacobi ≈
   μ·wsum/(1+μ·wsum) ≈ 33/34 ≈ 0.97`; 8 sweeps cut error by only ~22%. So the honest trade is **RB-GS
   (ρ≈0.5-ish at `mu=8.3`, fine in 8 sweeps) vs damped Jacobi (needs 20-40 sweeps)** — not "Jacobi is
   trivially better for GPU."
4. **The RB-GS "read-after-write hazard" is NOT a cross-arch hazard (scope-risk, numerical-methods).** With
   a **fixed two-color schedule**, red cells read only black and black read only red — there is *no
   intra-color RAW*, the read-set is exactly defined, and the result is identical on every architecture. It
   is a schedule you must pin, not a desync. On GPU it is two kernel launches with a barrier between colors —
   deterministic by construction.
5. **No shipped lockstep RTS runs Chebyshev in its deterministic core (lockstep-practitioner).** They ship
   the dumbest order-fixed solver that converges. Chebyshev is a convergence optimization masquerading as a
   determinism deliverable.

**Verdict:** the canonical solver stays **fixed-schedule Red-Black Gauss-Seidel**, reformulated in
**residual/flux form (§3.2)** with the **precomputed-reciprocal multiply (§3.1)**. Keep the float RB-GS as
the CPU reference oracle. The oracle chain is **float-GS → int-GS (arithmetic check) → (optional later)
any algorithm change (algorithm check)** — never validate a new algorithm directly against float-GS. The
draft's Zhu 2023 convergence citation was *over-extrapolated* (it is a Richardson iterative-refinement
result for analog/ReRAM inverse problems, not Jacobi-on-a-diffusion-stencil); the real convergence claim
rests on classical spectral-radius theory for the diagonally-dominant `(I−μΔ)` operator, settled
**empirically** by the S5 residual test (`atmosphere_solver.cpp:263-301` already measures the GS residual).
Iterative refinement is the named **escape hatch** if a field shows a visible quantization floor (§8.2) —
that is what Zhu actually demonstrates.

### 3.4 Reconciliation with `gs_precomputed_reciprocal_techniques.md`

| Note item | Status in this plan |
|---|---|
| Headline: divide isn't the villain; FMA + sum-order + RB read-after-write + `mean_wp` are | **Confirmed.** Keep as mental model. |
| Prereq: kill `/fp:fast`, pin contraction off | **Step zero (S0)** for the CPU reference + all float precompute. **Stronger than the note:** `/fp:precise` is NOT enough — use `/fp:strict` + explicit contraction-off on float TUs feeding integer state (M1). |
| Prereq: two-seed determinism harness first | **Unchanged and binding** — the gate for everything (§6). Extended to P1+P2 + Python state. |
| Technique #1 (precomputed `Dinv`) | **Promoted from optional to the S5 BASELINE** (the only shippable integer GS — §3.1). Caveats bind: explicit `continue`, rebuild keyed on `(mu \| obstacles \| is_wall \| is_vacuum \| permeability)`. |
| Technique #2 (integer-bucket LUT) | **Has NO determinism justification** once the divide is an integer reciprocal and the `min` is exact (lockstep-practitioner). The "9-entry table" claim is *wrong* — the face is `min(perm_i,perm_n)` and `wsum` is a sum of four such mins, so the key is `(self-perm, sorted-neighbor-tuple)`, not 9 buckets. **Pure throughput; strongly consider dropping** (Open Q4). The one piece worth pulling forward is the `w∈{0,½,1}→shift` collapse (M7) — a *correctness/rounding* simplification (collapses the 3-factor `mul·w·diff` product), into S5a. |
| Technique #3 (power-of-2 `mu` → pure shift) | Endgame; forces `d_atm≈36` (5.6× change) → separate physics-retune patch with visual A/B. Off the table unless `d_atm=200` is not load-bearing (Open Q3). |
| Rejected: Newton-Raphson reciprocal | **Stays rejected.** |

---

## 4. The deterministic global reduction for `mean_wp`

### 4.1 Why it's #1

`mean_wp` (`atmosphere_solver.cpp:143-151`) is a mean of `wave_p` over non-obstacle tiles, **subtracted
from every cell** (`:155-159`) — so any error contaminates the *entire* atmosphere field that tick. On GPU
a float sum's parenthesization differs across CPU-scalar / CPU-SIMD / CUDA-warp-shuffle, and float add is
non-associative (a parallel sum ≡ "a random permutation"; CCCL: atomics give "a different order between
runs"). **No amount of `-ffp-contract=off` + pinned CPU order fixes this cross-GPU in float** — it needs
RFA-class machinery.

### 4.2 The integer fix — order-free for free

Once `wave_p` is Q16.16 int32, the sum is a plain integer sum → **bit-identical regardless of order**.
We get CCCL's strictest GPU-to-GPU tier *without* the RFA cost.

**Accumulator width:** Q16.16 in int32 spans ±2³¹. Against N = 2²⁰ (1000×1000), worst-case `bits(sum) ≤
20+31 = 51` — 12 bits of int64 headroom (physical `wave_p` bound makes it ~37). **int64 accumulator, no
saturation.** `count` stays int32 (safe ≤ 2³¹).

```
int64_t sum = 0; int32_t count = 0;
for (...) if (interior(i)) { sum += wave_p[i]; count++; }   // membership = pure bool mask
int32_t mean_wp = (count > 0) ? (int32_t)(sum / count) : 0; // sum already Q16.16: NO pre-shift
```

**Two sharp edges the panel caught:**

1. **Scaling differs from the GS divide (M3).** `sum` is *already Q16.16*, so `sum/count` is Q16.16 with
   **no `<<16` pre-shift** — applying the general "pre-shift the numerator" rule here gives Q32.16 and a
   wrong mean. The GS divide pre-shifts; the mean does not. Two divides, two scalings — state both.
2. **Signed truncation bias (M3, numerical-methods).** `wave_p` is signed; `sum/count` truncates toward
   zero, so `mean_wp` is biased toward zero asymmetrically with the sign of `sum`, which flips across a
   trajectory → a sign-correlated DC drift injected into *every* cell. Deterministic (P1-fine) but a **P2
   conservation defect** the float version doesn't have. **Decision: round-to-nearest-even on the mean**
   (`(sum + (sign·count)/2)/count`, defined sign-symmetrically and pinned) to remove the bias. Forbid the
   compiler from magic-number-reciprocal'ing the runtime `/count` (§6.1).

### 4.3 GPU reduction pattern (for the later port)

Integer `atomicAdd` to an int64 accumulator **is** deterministic (associative) and avoids a multi-kernel
tree — *if* the membership `count` is a compile-time-derivable mask (function of obstacles/walls/vacuum,
**evaluated on integer fields only** — never a float-bridge `atmosphere < thresh` predicate, which would
inject a float comparison into the mask, M / research-grounding). Reserve the fixed warp-shuffle → block →
second-kernel tree for the harness/digest reference, not necessarily the shipping kernel. The real cost of
`mean_wp` on GPU is **the grid-wide barrier mid-tick** (wave → reduce → broadcast → subtract → diffuse),
not the adds — which is the performance argument for §4.5.

### 4.4 The integer cliffs are part of THIS migration (BLOCKER — three reviewers)

`n`, `n_smoke`, water `n` (`physics_engine.cpp:124-128,182-198,287`) are float→int `ceil` truncations over
float inputs (`max_dt`, and for `n_smoke` the `max_wind_sq` float reduction). Today they are deliberately
matched to numpy's double — a *same-binary* guarantee, **useless cross-GPU**. The CUDA port will not run
Python's double `ceil`. A 1-ULP slip flips an integer substep count → two peers run a different number of
iterations → instant unrecoverable desync, and a naive digest (bit-identical *within* a substep) misleads.

**Fix (first-class S0/S1 deliverable, NOT a §5 footnote):**
- `max_dt = 0.5/c` and the water CFL are config-constant → **precompute the substep counts at config-load**
  from config + a deterministic fixed-point CFL; OR compute `max_dt()` as a Q16.16 value from a
  deterministic fixed-point `sqrt` + integer div, with `ceil` as a fixed-point bit-op.
- `n_smoke` depends on the *runtime* `max_wind_sq` → compute it as an **integer max over the Q16.16 wind
  field** (max is order-free), then `d_eff_max` and the `ceil` in fixed-point; or cap `n_smoke` at a
  documented config max.
- The harness must drive a config where **each cliff exceeds 1** and assert the integer count is identical
  across the cross-config matrix (§6.2.5) — impossible to satisfy honestly while the inputs are float.

### 4.5 Retire the global reduction — the recommended target, not an "appetite question"

The mean exists only to keep the transfer mass-neutral. A **local edge-flux conservative transfer**
(mirroring temperature conduction, but in the R2-CONS once-per-face form) is conservative *by construction*
in integer arithmetic, removing the global reduction (and its barrier), the truncation DC-bias (§4.2), and
the deepest hazard — all at once, GPU-local. The panel (numerical-methods, determinism-adversary,
gpu-perf-skeptic) converged: **this is the same refactor R2-CONS forces you toward anyway** (§2.3), and
the global barrier — not the adds — is the GPU throughput villain. **Ship the rounded int64 mean now (S3)
as the deterministic, obviously-correct stopgap; plan the edge-flux transfer as the S-step that retires
`mean_wp`** (Open Q8 is reframed from "should we?" to "confirm priority"). You may never need the mean.

### 4.6 The other reductions

`max_wind_sq` (feeds the `n_smoke` cliff), `d_smoke_max`, fire `max_fire` early-exit, the
`boiling.any()`/per-plane `.any()` booleans. Max and any are order-independent for integers too — but
**every membership predicate must be evaluated on integer fields** (§4.3), and the cliffs (§4.4) are where a
1-ULP slip flips an integer.

---

## 5. Transcendentals per system

### 5.1 The principle

For determinism you want the most *reproducible* transcendental, not the most *accurate*. A 2-ULP `expf`
that wanders across architectures is useless; a committed LUT off by a fixed known amount that returns the
identical integer everywhere is exactly what lockstep needs. Accuracy is a game-feel budget (Q16.16
δ≈1.5e-5 → any approximation good to the bottom 2-3 frac bits is invisible), fully decoupled from
determinism. **Pick ONE canonical integer algorithm per function and freeze it** — a LUT and a CORDIC give
different last bits; you cannot mix CPU-LUT with device-CORDIC.

### 5.2 Sim-path inventory (audited from the code — and the divides the draft missed)

**REAL transcendentals on the sim-state path:**

| Op | Site | Recommendation |
|---|---|---|
| `sqrt` — fire wind-magnitude `W` | `fire_simulation.cpp:84` | **Fixed-iteration-count** integer-Newton / digit-recurrence sqrt (FixedMath/FixPointCS algorithm). Fixed count (not `while converged`) → branch-identical across all lanes (IDEA 12). |
| `sin`/`cos` — ray directions | `raycaster.cpp:139-140` (feeds Q16.16 `heat` → sim state) | **Committed Q16.16 sin/cos LUT shipped as a versioned data file** + integer lerp; ray angles are a deterministic function of source params, but **audit `LightSource.jitter` (`raycaster.h:55`)** — if jitter perturbs the angle per-frame via RNG, the jittered angle is sim state and still needs the deterministic LUT, not a per-source precompute (m4). |
| `sqrt`+`floor`/`round` — smoke back-trace step count + bilinear indices | `smoke_dynamics.cpp:63-64,72-73,95-99` | fixed-iter integer sqrt; **enumerate the THREE rounding modes** the back-trace uses — `floor` (`:95`), `floor(x+0.5)` round-half-up (`:72-73`), and the bilinear renorm divide `acc/wsum` (`:122`, a *second* dynamic divide, M8) — and map each to its exact integer idiom (m-rounding). A mask is `floor` toward −∞; the back-trace does not only floor. |
| `sqrt` (CFL), `tan` (tilt) — water | `water_solver.cpp:10,17,56-57` | **Scalar, once-per-tick / once-per-config** precompute. But the CFL `sqrt` feeds the `n` cliff (§4.4) → it must be the **same deterministic fixed-point routine** on every peer, not a host `std::sqrt`. |

**Divide-by-config constants the draft's §5 audit MISSED (M7, m7):** fire's `smoothstep(P_min,P_full,P)`
and `(T−fire_T_ext)/fire_T_span` (`fire_simulation.cpp:87-88`), `fuel_ref`/`p_expand_ref` normalizations,
`atmosphere/p_expand_ref` (`:111`); smoke and water have their own. **Re-audit every divide (not just
transcendentals) in fire/smoke/water** — each is a precompute-reciprocal-at-load candidate (the temperature
`heat_inv_shift`/`face_shift` pattern) and each is a place a stray runtime float divide can hide.

**FLOAT-EXEMPT (render-only):** `exp(-tau)` on `remaining[]` RGB (`raycaster.cpp:303-305`),
scatter→`smoke_glow`, `1/sqrt` direction normalize, cosine angular falloff *iff* it only touches
`light_rgb`/intensity. **Audit during impl** whether the cone falloff leaks into `heat_emit` — if it does,
it's sim-side.

### 5.3 Library + CUDA discipline

**FixPointCS (or a C++ port) is the CPU oracle** — purpose-built for bit-exact cross-platform
sqrt/sin/cos/exp (a better oracle than `fpm`/`libfixmath`, which do not promise cross-platform bit-exactness).
Each hot function is re-expressed as `__device__` integer code for the GPU; the contract is **same integer
algorithm both sides → bit-identical by construction** — but that is *necessary, not sufficient*: pin
iteration counts, avoid 128-bit intermediates, and **where a LUT is chosen, ship the table as a committed
versioned data file** (CPU and device read identical bytes) with a unit test `for all int32 in-range:
cpu_op(x) == golden[x]`. The digest test (§6.2.6) catches any CPU/device divergence the moment the CUDA
path exists.

### 5.4 Flag for later

If radiative `T⁴` cooling or Beer-Lambert *heat* attenuation ever becomes sim state → Q16.16 LUT+lerp then.
Not today.

---

## 6. Migration ORDER + per-step gating + the harness

### 6.1 Step 0 — the prerequisite gate (precedes ALL field work)

1. **Pin `-std=c++20` host + device** in CMake with a `static_assert` (the shift/division determinism
   argument depends on it — research-grounding).
2. **Kill fast-math with the right per-compiler flags (M1, lockstep-practitioner).** `/fp:precise` is NOT
   enough — MSVC still contracts FMA under it. For *every* TU touching float on a path feeding integer
   state (load-time bakes, residual float bridges): MSVC `/fp:strict` + explicit contraction-off; GCC/Clang
   `-ffp-contract=off -fno-fast-math`. Compute all load-time bakes in **double under `/fp:strict`** and
   **hash the baked integer cache into the version stamp** so two peers with divergent bakes fail loudly.
   Forbid magic-number-reciprocal lowering of the runtime `/count` and GS reciprocal.
3. **Close the determinism contract past C++ into Python (BLOCKER — scope-risk).** `combat.py:200-247`
   reads the integer `heat` field and does **float double** damage math → `u.current_hp -= dmg` →
   **death decision** `if u.current_hp <= 0`. Unit HP and hit/kill events are lockstep-critical synced
   state. The harness hashes `gmap` fields but **never hashes `current_hp` or the event stream** — it would
   stay green while two peers diverge on who lives and dies. **Either** move unit damage/ignition into the
   C++ integer path, **or** make `dmg`/`current_hp` pinned-integer fixed-point in Python, **or** explicitly
   document "all peers run identical CPython+NumPy on x86" as the contract (which contradicts cross-GPU
   cross-vendor lockstep). `apply_temperature_ignition` is already an integer-threshold compare (good — keep
   that pattern). **Extend the harness/digest to hash ALL synced state, including unit HP and events.**
4. **Build the two CPU-only falsifiable tests (P1 + P2 — determinism-adversary, scope-risk):**
   - **Reduction-permutation test (P1):** compute every reduction (`mean_wp`, `max_wind_sq`, …) via forward
     scan, reverse scan, randomized pairwise tree, and a 4-wide SIMD-like tree; assert all **bit-identical**.
     For integer this must pass; it fails the instant a non-associative step (premature narrowing, a float
     bridge, a `/` before sum) sneaks in — *on the CPU, before CUDA*. Make it a permanent property-based
     test over random integer fields.
   - **Sealed-room conservation test (P2):** assert `Σ atmosphere` over a closed region (and `Σ smoke`, `Σ
     water`) is constant to the LSB across a long settle. Catches the R2-CONS / truncation-leak class that
     P1 is blind to.
5. **Desync canary (IDEA — scope-risk):** a CI job that compiles one TU with `/fp:fast` *on* (or perturbs
   one reduction order) and asserts the harness goes **red**. A green-only harness never seen to fail is not
   yet known to work.

### 6.2 The harness design (extends `tests/field_ab_harness.py`)

The existing harness is the right granularity (per-cell, per-field, per-tick, `tol=0.0`, locates the worst
cell) but is **single-machine** — necessary, not sufficient (it cannot prove cross-GPU). Add a driver on
top (reuse `_snapshot` + `diff_trajectories(tol=0.0)`):

1. **Three field buckets, not two (m5).** Q16.16 sim fields (digest, `tol=0.0`), bool topology masks
   (`obstacles`/`is_vacuum`/`material` — 1 byte, serialized distinctly), float render (excluded). **Drop
   render-only `ripple`/`ripple_v` from the fixed-point digest's `SIM_FIELDS`** (`field_ab_harness.py:54`)
   — but scope the removal to the *new* digest; do not silently shrink the legacy Patch-1 refactor harness's
   set (scope-risk m). Verify `dyn_light_atten` is render-only too.
2. **Within-config self-match:** two runs → bit-identical (uninitialized scratch, RNG leak, order-dependence).
3. **Seed-independence (no state-leak):** seed-A, seed-B, seed-A; second A must match first (catches solver
   `mutable` scratch bleed — `vac_dist_`/`lap_`/`rhs_`/`scratch_`).
4. **Cross-config:** vary the params that move integer cliffs and quantization — `ticks_per_second` (every
   `ceil` cliff), `wave_c`/`d_atm`/`k_p` (`mu`, CFL, substep counts), `tile_size_m`/`dx` (water SI scale).
   Each config self-matches at `tol=0.0`.
5. **Assert dangerous paths are exercised** (else green is vacuous): `mean_wp` hit; `max_fire`/`.any()`
   flips; **all three integer cliffs take a value >1 in ≥1 config**; a breach opens mid-trajectory
   (`destroy_wall(8,0)`) so `is_vacuum`/adjacency change and the Dinv rebuild + vacuum-cooling flip are
   tested; **a pathological-stress scenario drives each saturating field past its clamp** (proves both peers
   clamp at the same tick) and each accumulator toward its worst-lifetime bound (debug overflow trap fires
   if exceeded — m).
6. **Canonical per-tick digest (cross-machine / CUDA hook):** per-tick blake2/sha256 over the concatenated
   field bytes in a **committed, versioned serialization spec** (a `.toml`: field → index → dtype → shape →
   endianness, frozen before any golden is recorded). Report **per-field** so a reduction desync names
   itself; **on mismatch, fall back to the per-cell locator** (first-divergent field/cell/tick) — the
   needle-in-2²⁰-cells problem otherwise (research-grounding, lockstep-practitioner). **Scratch buffers must
   be fully written or zero-filled every tick** (poison with a sentinel in debug, assert none survives into
   an output) — CUDA `cudaMalloc` is not zeroed, so a read-before-write padded lane reads device-specific
   garbage the CPU harness is blind to (M5).
7. **Golden persistence with a version stamp:** records config hash, seed, **Q-format version tag** (incl.
   per-field GPU widths §2.6 + the serialization spec hash + the load-time-bake hash), build/arch id. Golden
   regen MUST be in the same commit as the arithmetic change; CI rejects a stale-tag golden.

### 6.3 Per-step migration — ordered by COUPLING DIRECTION (migrate producers before consumers)

The draft's "risk-first, water-first" order maximizes float-bridge surface: water couples to atmosphere
(W3) and smoke (W5 steam), and smoke advects on wind = ∇atmosphere — so the two "isolated" first systems
both bridge *to* the system migrated last (lockstep-practitioner, scope-risk). **A migrated system reading
through a float-bridge from a not-yet-integer neighbour is NOT cross-GPU-deterministic** — the bridge
dequantizes int→float, runs a cross-GPU float op, requantizes (M6). So the honest dependency DAG drives the
order, and we migrate in **coupling-closed groups**:

- **S0** — the prerequisite gate (§6.1). Gate for everything.
- **Spike 0 (one week, BEFORE the multi-week migration — BLOCKER, scope-risk):** implement the `mean_wp`
  int64 reduction as a CPU reference *and* a throwaway CUDA kernel; on real hardware demonstrate (a) the
  integer digests match byte-for-byte across ≥2 CUDA architectures (cloud T4/A100/L4 by the hour) — or at
  minimum CUDA-int == CPU-int — and (b) the **float version FAILS the same test**. This validates the
  load-bearing premise ("integer is bit-identical across GPU arch, float is not") *before* the budget is
  spent, instead of discovering a surprise at the very end in the CUDA port.
- **S1 — Water** (cleanest range/SI, only scalar `sqrt`/`tan`). Lands the per-field idioms (R2-CONS for
  `water_depth`, sign-symmetric flow velocity) + the water `n` cliff in fixed-point. **Documented:
  self-reproducible (P1-same-machine) from here, but NOT cross-GPU-deterministic until its W3/W5 atmosphere
  + steam couplings are integer** (S3/S5).
- **S2 — Atmosphere/wave/wind/smoke/gas as ONE coupling-closed group** (they are tightly coupled via
  `mean_wp`/wind/diffusion — migrating them together eliminates the float bridges *within* the group). This
  is the determinism centerpiece and everything downstream reads atmosphere/wind. Sub-steps, each gated:
  - **S2a** — wave + `mean_wp` (int64 rounded mean, §4.2) + the `c_sq*lap` int64 dt-order discipline (M5);
    the reduction-permutation test now bites.
  - **S2b** — smoke + 5 gas planes (R2-CONS; the semi-Lagrangian back-trace — fixed-iter integer sqrt, the
    three rounding modes, the bilinear renorm divide M8) + `n_smoke` cliff in fixed-point.
  - **S2c** — atmosphere diffusion: **integer Red-Black GS, residual/flux form, precomputed-reciprocal
    multiply** (§3.1-3.3) with the `w∈{0,½,1}→shift` collapse (M7); + wind (∇atmosphere, 2-term diff `>>1`).
    Behavior change vs float golden (truncating reciprocal ≠ IEEE) → regen golden; **acceptance =
    self-reproduction at `tol=0.0` AND the integer GS Linf residual within a stated factor of the float
    build's** on the stress scenarios (`atmosphere_solver.cpp:263-301`). "Deterministic" and "converges" are
    separate claims — test both (scope-risk).
- **S3 — Fire** (most cross-coupled: reads temperature/atmosphere/wind, writes fire/smoke/atmosphere/wall_hp;
  needs all upstream integer → why it's late). The 6-factor logistic in Q16.48 (M2); `smoothstep`/`clamp01`
  pinned; `temperature*inv_temp` → Q-quantized threshold; the `wall_hp<=0` burn-through list is a
  control-flow output that must be bit-deterministic.
- **S4 — Cleanup:** delete float-bridge seams; the **per-TU "no float/double/`/fp:fast`" CI ratchet**
  (added incrementally as each TU lands, S0 onward — NOT only at the end — so a later patch can't reintroduce
  float into a migrated solver, IDEA lockstep-practitioner). Mind the legit exceptions: the load-time bake TU
  and render buffers.
- **Later (separate patches, feel-gated, NOT on the determinism critical path):** retire `mean_wp` via
  edge-flux (§4.5); Technique #3 `d_atm` retune; 16-bit gas/smoke/fire on GPU (§2.6); the
  cross-architecture CI gate on ≥2 physical NVIDIA GPUs (the only real cross-GPU proof — until it runs, all
  green harness results are labelled "single-machine determinism proven; cross-GPU UNVERIFIED").

**Per-step gating contract:** every step lands with (1) within-config self-match `tol=0.0` for all configs,
(2) no state-leak across seeds, (3) the P2 conservation test green for conserved fields, (4) golden
regenerated in the same commit with a bumped version tag, (5) dangerous paths asserted-exercised. A step
that changes behavior vs the float golden (the integer GS, any retune) is labelled **feel-gated** and gets a
**committed-artifact A/B** (see §6.4), not a live eyeball.

### 6.4 Feel-regression harness (distinct from the determinism harness — BLOCKER scope-risk)

`tol=0.0` answers "is it deterministic?" — **nothing** about "does it still look right after quantization?"
Specify a separate feel-regression gate so a slow drift (a field resting 1.5e-5 low, fire igniting one tick
late from a Q-quantized threshold) is caught, not eyeballed:

- Scripted scenarios (the `default_scenario_sim` + firestorm + flood + blast) to N ticks under (i) the float
  golden and (ii) the integer build, compared with a **physical tolerance** (e.g. `atol≈2e-4` on normalized
  fields, max-cell and L2) — here the two sides *should* differ slightly; assert the difference is small
  **and the mean signed error ≈ 0** (unbiased — directly tests for the R2-CONS / truncation leak in
  field-feel space).
- For genuinely-different steps (the integer GS), render N frames to PNG from both, diff (SSIM / per-pixel
  with a numeric threshold), commit as an artifact Erik reviews.
- **Verify "invisible" post-render, not in field space:** a rendered smoke density goes through a tone curve
  and accumulates over a column, so confirm δ≈1.5e-5 stays invisible after the render path.

### 6.5 Live desync detection + forensics (BLOCKER — lockstep-practitioner)

Every shipped fixed-point lockstep post-mortem says the same thing: **you will desync during development;
what saves the project is fast detection + bisection to the offending tick/cell.** The plan has excellent
*prevention* and a good offline digest; it must also have the operational loop:
- **Online per-tick (or per-N-tick) rolling checksum** exchanged between peers in the netcode path (reuse
  the §6.2.6 digest) — a desync is caught at tick 1043, not "the game looks weird 5 minutes in."
- **Forensic dump on mismatch:** both peers dump full int32 field state (+ unit/event state) for the
  diverging tick to disk → diff which field/cell first diverged.
- **Input-log replay:** record input stream + seed + Q-format version so any desync is reproducible offline.

### 6.6 Rollback story (corrected — scope-risk)

The draft's per-system `USE_FIXED_POINT` runtime switch is dropped — maintaining dual float+integer paths
(three for atmosphere) is a large unbudgeted burden teams don't actually carry. The real rollback primitives
we already have: each S-step is one commit; the float-bridge keeps unmigrated neighbours on float; **`git
revert` of a step reverts one field family.** Keep the float CPU reference solver alive **only** as a
separate reference function compiled into the *test* build (the harness oracle, e.g. float RB-GS), not as a
shipped `#ifdef`. The temperature solver is the safety proof: it shipped in this exact style with the
per-cell harness green — the migration is "make the other systems look like temperature (with R2-CONS for
the conserved ones)," not invent a new technique.

---

## 7. Fold the gs_precomputed work in, or after? — verdict

**Pull the prerequisite forward into S0; fold Technique #1 INTO the S2c GS port (not a later patch); defer
#2/#3.**

- **S0 (now):** the fast-math kill — the gs note's own Prerequisite #2, now sharpened to `/fp:strict` +
  contraction-off on float TUs feeding integer state (M1), `-std=c++20`, and the load-time-bake hash. The
  integer port depends on it for the residual float precompute passes. Non-negotiable.
- **Technique #1 → INTO S2c (changed from the draft).** The draft deferred it to "later performance." The
  gpu-perf-skeptic shows the precomputed-reciprocal `Dinv` multiply is **the only shippable integer
  formulation of the GS solve** (a per-cell int64 divide is the worst op on Ampere). So it is not an
  optimization to schedule later — it is the baseline S2c is validated against from day one (§3.1). Because
  we keep RB-GS (not Jacobi), there is no separate Jacobi reformulation step for it to ride on; it folds
  directly into the integer GS port. Pull the `w∈{0,½,1}→shift` collapse (M7) forward too — it's a
  rounding/correctness simplification, not just throughput.
- **Technique #2 (integer-bucket LUT) — strongly consider DROPPING.** Once the divide is an integer
  reciprocal and the `min`-of-pairs face is exact, #2 has **no determinism justification** (only throughput),
  and its key is *not* the claimed 9-entry table (it's `(self-perm, sorted-neighbor-tuple)`). Carrying the
  perm-quantum invariant machinery is unjustified unless permeability leaves `{0,0.5,1.0}` (Open Q4).
- **Technique #3 (power-of-2 `mu` shift) — endgame, separate physics-retune patch** with visual A/B; off the
  table unless `d_atm=200` is not load-bearing (Open Q3).

**The gs doc folds in PARTIALLY and at S2c, not as a separate later patch:** its prerequisites are S0, its
Technique #1 is the S2c baseline, its #2/#3 stay out. **After this migration ships, the gs doc is
superseded** — its open questions are answered here (the contract is cross-GPU, §1; #1 is mandatory not
optional; #2 likely dropped; #3 is the endgame) — and it should be archived with a pointer to this plan.

---

## 8. Risks, precision / game-feel, and the Ampere datapath budget

### 8.1 Risks

- **Overflow / signed-UB (the main new burden).** Per-field range budgeting (§2.4) is an *argument*, not a
  proof, until the harness drives pathological inputs. **Mitigation:** declared overflow policy per field
  (§2.4), int64 intermediates with dt-before-narrow (M5), worst-*lifetime* bounds (op bound × max
  substep/iter count, not single-op — scope-risk m), saturation applied **identically on the int64
  intermediate AND the int32 store** (lockstep-practitioner), and a **debug-build assert-on-overflow at every
  int64→int32 narrow** running in the stress scenarios (catch a wrap loudly in CI, never silently in play).
- **The conserved-field truncation leak (R2-CONS).** Deterministic-but-wrong mass drift that stays green on
  P1 forever. **Mitigation:** the P2 conservation test (§6.1) + the edge-flux idiom + the feel-harness
  mean-signed-error check (§6.4).
- **Transcendental drift between the CPU oracle and the `__device__` version.** **Mitigation:** same integer
  algorithm both sides, fixed iteration counts, committed LUT data files, the digest test (§6.2.6).
- **The integer cliffs (§4.4)** flipping a substep count cross-GPU — addressed as a first-class deliverable.
- **Golden churn / version confusion / stale golden / render-buffer leak** — the version tag, same-commit
  regen, CI stale-tag reject, the three-bucket digest.
- **Python contract leak** (§6.1) — extend the digest to unit HP/events; fence or migrate `combat.py`.

### 8.2 Precision / game-feel

- **Iterative-stall floor:** with 16 frac bits on a 0..1 field the resting granularity is ~1.5e-5 —
  invisible, and the difference/flux idioms make the rest state exact (a feature, like temperature's
  dead-band). If a field shows a visible floor, the remedy ladder is: (a) accumulate fluxes in int64, apply
  once [already R2/R2-CONS]; (b) **iterative refinement** — compute the residual in the same fixed-point,
  solve the correction, add it back (what Zhu 2023 actually demonstrates — recovers precision beyond the LSB
  without widening); (c) rescale that field to more frac bits; (d) only if measured, Q8.24 for that one
  field. **Measurement drives any per-field split.**
- **The `d_atm≈36` retune** (Technique #3) is a real 5.6× physics change — explicitly out of scope unless
  ratified.

### 8.3 The Ampere datapath budget (the draft's perf blind spot — gpu-perf-skeptic)

State the cost of admission up front: the RTX 3070 (GA104) SM has **16 INT32 cores vs 32 FP32 cores**, and
going all-integer **halves peak arithmetic throughput** and gives up the concurrent INT32/FP32 dual-issue
Ampere was built around. Every `mul64` is ~3 INT32 ops on the half-rate datapath. The optimistic case rests
on the workload being **memory-bound** (a diffusion stencil on a 448 GB/s part), so the arithmetic slowdown
partly hides behind DRAM latency — **but only if** (a) no per-cell int64 divide survives (§3.1), (b) int64
intermediates are minimized (R1′), and (c) the [0,1] fields go int16 (§2.6, halving the dominant gas-plane
traffic). **Add a roofline checkpoint to S2c (IDEA):** profile the integer diffusion kernel's
arithmetic-intensity; if it lands compute-bound after going integer, escalate the reciprocal-multiply +
16-bit packing *before* the CUDA port, not after. int64 intermediates also raise register pressure → watch
occupancy.

---

## 9. OPEN QUESTIONS for Erik (decisions only he can make)

1. **Determinism contract scope — confirm "any-machine, cross-GPU" is final** (D-C in the gs note). The
   locked decision says yes; this plan assumes it absolutely (it's why the cliffs §4.4, the Python fence
   §6.1, and the Spike 0 GPU test exist). Confirm so we never relitigate.
   *Recommendation: confirm yes (it is already locked); this question exists only to bind §6.1's Python-fence
   scope, which is the surprising consequence.*
2. **The Python sim-state consumers (`combat.py` damage/kill math) — which fence? (NEW, BLOCKER).** Move
   unit damage/ignition into the C++ integer path, make `current_hp`/`dmg` pinned-integer fixed-point in
   Python, or document "identical CPython+NumPy x86 only" (which contradicts cross-vendor lockstep)?
   *Recommendation: move it to C++ integer over time (it already reads the integer `heat` field; the C++
   path is where the rest of the synced state lives); short-term, fence + hash unit HP/events into the
   digest so the leak is at least visible.*
3. **`d_atm` — tuned or default (200)?** Determines whether Technique #3's power-of-2 `mu` pin
   (`d_atm≈36`, 5.6× change) is ever on the table.
   *Recommendation: assume 200 is load-bearing → #3 off the table; stop at the integer RB-GS +
   precomputed-reciprocal. Revisit only if a profiling/feel campaign wants it.*
4. **Will permeability ever leave the `{0,0.5,1.0}` alphabet?** (continuous per-material perm, damage decay,
   soot occlusion). Drives whether Technique #2 + the config-load quantization invariant is worth building.
   *Recommendation: keep continuous Q16.16 permeability and DROP Technique #2 — it has no determinism
   justification once the divide is an integer reciprocal and the `min` is exact (the integer reciprocal
   already gives bit-exactness). Only revisit #2 as pure throughput if profiling demands it.*
5. **Smoke/gas/fire 16-bit on GPU: confirm freeze-the-width-now, ship-int32-on-CPU?** The draft deferred
   16-bit entirely; the panel argues freezing the per-field GPU width now (§2.6) avoids a second
   golden-churn cutover through every coupling seam.
   *Recommendation: yes — record int16 (Q1.15) for the [0,1] fields in the version tag now; storage stays
   int32 on CPU; the actual 16-bit kernels land in the later GPU patch. Zero determinism cost.*
6. **GS solver — confirm: keep fixed-schedule Red-Black GS, DROP Chebyshev-Jacobi?** This is the plan's
   biggest reversal from the draft. RB-GS converges in 8 sweeps at `mu=8.3` and its read-after-write is fully
   determinized by the fixed two-color schedule; Chebyshev re-imports a runtime spectral estimate (a global
   reduction) and is more quantization-sensitive.
   *Recommendation: confirm — keep RB-GS (residual/flux form, precomputed reciprocal). It unifies nothing to
   force Jacobi, and it costs determinism. (The draft's "unify with temperature via Jacobi" argument does not
   survive: temperature is a relaxation field, atmosphere is conserved — they get different idioms anyway,
   R2 vs R2-CONS.)*
7. **The one-week Spike 0 before the multi-week migration — approved?** Implement `mean_wp` as CPU + CUDA,
   prove integer matches across ≥2 GPU archs and float fails, before spending the field-migration budget.
   *Recommendation: yes — it is days, and it validates the single load-bearing premise on real hardware
   instead of at the end of the project where a surprise is most expensive.*
8. **Retire `mean_wp` via local edge-flux transfer — confirm it's the planned target (not just an
   appetite)?** It deletes the deepest reduction + its GPU barrier + the truncation DC-bias, and is the same
   refactor R2-CONS forces anyway. It is a (small) physics change.
   *Recommendation: ship the rounded int64 mean now (S2a) as the deterministic stopgap; plan the edge-flux
   transfer as a near-term S-step gated on the P2 conservation test. Confirm priority, not whether.*
9. **Coupling-group order vs strict risk-first.** This plan migrates atmosphere/wave/wind/smoke/gas as one
   coupling-closed group (S2) to avoid float bridges within it, with water first (S1) and fire last (S3). The
   alternative is strict easy→hard per-field with more (same-machine-only) bridges.
   *Recommendation: the coupling-group order — it is the only way any field reaches *cross-GPU* determinism
   before S4, and it makes the "deterministic" claims honest per step.*

---

*Files grounding this plan (verified at HEAD; re-verify line numbers before impl — Patch 1/2 moved this
code):* template — `cpp/src/temperature_solver.{h,cpp}` (R1/R2 idioms `:71-91`, sign-symmetric shift `:142`,
max-principle contract `.h:34-37`), `cpp/src/raycaster.h:25-45` (`HEAT_SCALE`/quantize/saturating-add).
Hardest systems — `cpp/src/atmosphere_solver.cpp:143-159` (`mean_wp` + signed transfer), `:257` (GS dynamic
divide / 3-factor `mu·w·atm` product), `:107-108` (wave_v integration / `c_sq*lap`), `:199-261` (RB-GS
sweep), `:263-301` (GS-residual hook), `cpp/src/smoke_dynamics.cpp:63-99,122` (back-trace
sqrt/floor/round-half + the bilinear renorm divide), `cpp/src/fire_simulation.cpp:84` (wind-mag sqrt),
`:87-95` (6-factor logistic + smoothstep + config-divides), `:144-145` (clamp01). Orchestration cliffs —
`cpp/src/physics_engine.cpp:124-128` (`n`), `:182-198` (`max_wind_sq`→`n_smoke`), `:287` (water `n`).
Python contract leak — `src/simulation/combat.py:200-247` (float damage/kill on integer `heat`), `:253+`
(integer-threshold ignition, the good pattern). Harness base — `tests/field_ab_harness.py:51-57,111-139`.
Build — `cpp/CMakeLists.txt:14,16` (`/fp:fast` to kill), `:42-45` (`/fp:precise` ≠ deterministic float).
Canon — `docs/gs_precomputed_reciprocal_techniques.md`, `docs/resolution_architecture_proposal.md`,
`docs/architecture/engine/02_state_and_ownership.md`.

*Research foundation:* NVIDIA, *Controlling Floating-Point Determinism in CCCL* (determinism tiers; integer
reductions avoid the RFA's 20-30%); NVIDIA, *Floating Point and IEEE 754* (`÷`/`sqrt` correctly-rounded
everywhere; `sin/cos`=2 ULP, `pow`=4 ULP, "may differ across architectures"); ACCU Overload 100, *Why Fixed
Point Won't Cure Your Floating Point Blues*; AMBER-SPFP (wide fixed-point accumulators for reproducible
force sums); FixPointCS / Photon Quantum (current shipped fixed-point-sim + float-render architecture and
bit-exact cross-platform transcendental library); yal.cc *Preparing your game for deterministic netcode* &
SnapNet *Netcode Architectures: Lockstep* (checksum-every-tick + forensic-dump discipline; no third-party
float / host sqrt in the deterministic path); Golub & Varga / Saad *Iterative Methods* (spectral-radius
convergence for the diagonally-dominant `(I−μΔ)`); Zhu et al. 2023, *Sci. Rep.* (fixed-point iterative
refinement recovers precision beyond the LSB — cited for what it actually shows, as the escape-hatch §8.2,
NOT as a Jacobi-convergence proof); jfdube, *Trigonometric Look-Up Tables Revisited* (Q16.16 LUT + integer
lerp). Watch-list (do-not-bet): JGS2, arXiv:2506.06494, 2025 — near-2nd-order GPU Jacobi/GS,
elastodynamics-specific.
