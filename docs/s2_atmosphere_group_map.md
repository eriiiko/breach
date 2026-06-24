# S2 — Atmosphere / Wave / Wind / Smoke / Gas coupling group — fixed-point (Q16.16) MAP

**Status:** MAP (read-only survey), 2026-06-24. Input to the S2 plan (the next session writes that).
Companion to `docs/fixed_point_migration_plan.md` (the approach + §9 locked resolutions) and
`docs/s1_water_fixed_point_plan.md` + `cpp/src/fixed_point.h` (the S1 toolkit + the conservative-flux
pattern to REUSE). This document is **descriptive of HEAD**, not prescriptive — it catalogues every divide,
transcendental, reduction, comparison, conservation property, and integer cliff in the five systems that
migrate **together** (so the float bridges *within* the group are eliminated in one move).

Source files surveyed (verified at HEAD):
- `cpp/src/atmosphere_solver.cpp` / `.h` — wave_substep + diffuse_solve (wave kick, mean_wp, RB-GS, BCs, wind)
- `cpp/src/smoke_dynamics.cpp` — semi-Lagrangian advection, wind-diffusion, sink_hop
- `cpp/src/physics_engine.cpp:111-252` (`run_substeps`) — the n_wave / n_smoke cliffs, the per-gas loop
- `src/simulation/gamemap.py:83-123` — field dtypes/shapes; `src/simulation/gases.py` — N_GASES=5
- `config.toml [physics]` / `[gases.*]` — the shipped constants
- `tests/field_ab_harness.py:71-74` — the SIM_FIELDS digest set

> **One-line orientation.** The group is a producer→consumer chain per tick:
> **wave** (explicit, `n_wave`×) → **mean_wp** (global reduction) → atmosphere transfer →
> **atmosphere diffusion** (implicit RB-GS, ×1) → **wind = −∇(atm+wave_p)** → **smoke/gas advection**
> (semi-Lagrangian, `n_smoke`× on the once-computed wind) → **sink_hop** (K×). The plan's sub-step
> order **S2a (wave + mean_wp) → S2b (smoke + gas) → S2c (atmosphere diffusion/GS + wind)** follows this
> data flow; note wind is *produced* in S2c but *consumed* in S2b, so the group must land atomically
> (within-group float bridges only close when all of S2a/b/c are integer — see §6).

---

## 1. Fields — dtype, shape, range, Q16.16 sizing

All fields are row-major `(h, w)` **float32** today (declared `gamemap.py:86-123`), passed to C++ as raw
pointers across pybind. `gas` is the one exception: a dense **`(N_GASES=5, h, w)` float32** contiguous
array (`gamemap.py:99`), and `smoke` is a **view into `gas[BLACK_SMOKE]`** (`gamemap.py:109`) — so "smoke"
and "the 5 gas planes" are the *same* storage; black_smoke IS plane 1 of 5. `N_GASES = 5` (`gases.py`,
order: white_smoke, black_smoke, poison, teargas, fuel_gas).

| Field | Shape / dtype today | Realistic range | Signed | Conserved? | Q16.16 sizing notes | GPU width (frozen, plan §2.6 / Q5) |
|---|---|---|---|---|---|---|
| **atmosphere** (pressure) | (h,w) f32 | ~1.0 interior; →0 drained vacuum; fire plume up to ~2.0; W3 displacement ×[1/1.5, 1.5] | no | **YES** | Comfortable in ±32768. Smallest meaningful increment is the late transfer `xfer·(wave_p−mean_wp)` ~1e-3 → ~65 counts; fine. | **int32** |
| **wave_p** (acoustic anomaly) | (h,w) f32 | zero-mean; source feed 8-10 (`max_source_per_step=10`, `feed_rate=200`); rings then decays | **YES** | mass-neutral via mean_wp | Comfortable. Q24.8 *candidate* only if `wave_v` forces it (§2.4 plan). | int32 (Q24.8 candidate) |
| **wave_v** (wave velocity) | (h,w) f32 | tens→hundreds in a blast | **YES** | no | **OVERFLOW WATCH.** `c_sq·lap` with `c=66 → c_sq=4356` (config; the plan's table assumed 300/4356 — re-derive: at `wave_c=66`, `c_sq=4356` still, the constant is squared) times a lap O(10) ≈ 4e4 — **exceeds ±32768 BEFORE ·dt**. Safe ONLY if the int64 intermediate carries `c_sq·lap` and `·dt` is applied *before* narrowing (plan M5). Measure peak \|wave_v\|; most likely Q24.8 exception. | int32 or Q24.8 (measure) |
| **wave_source** | (h,w) f32 | injected energy, 0..~10 | no | no (drained into wave_p) | Comfortable. | int32 |
| **wind_x / wind_y** | (h,w) f32 each | gradient O(0.1-1); shockwave spikes higher | **YES** | no | 2-term central diff ×0.5 (`>>1`). Feeds smoke advection AND the `max_wind_sq` cliff → **must share format with smoke's reads**. | int32 |
| **gas** (5 planes incl. smoke/black_smoke) | **(5,h,w) f32** | **[0,1] clamped** (`smoke_dynamics.cpp:213,272`) | no | **YES** | Bilinear-renorm divide keeps it normalized. Densities sub-1 → late increments sub-LSB at int16; comfortable at int32. The widest sim memory (5 planes). | **int16 (Q1.15)** — the dominant bandwidth win |
| **smoke** | view of gas[BLACK_SMOKE] | same as gas | no | YES | (same storage as a gas plane) | int16 (Q1.15) |

**int16-on-GPU candidates (Q5 locked):** the **[0,1]-clamped** fields — **all 5 gas planes (incl. smoke)**.
The physical fields (atmosphere/wave_p/wave_v/wave_source/wind) stay **int32** (signed and/or >1 range).
Ship int32 on CPU; record int16 (Q1.15) for the gas planes in the format-version tag now (§2.6) so the
CUDA buffers + digest schema are designed once. int16 add/mul stay associative/exact → zero determinism cost.

---

## 2. Per-system arithmetic — every divide / transcendental / accumulation / comparison

Flags: **[÷]** dynamic divide, **[÷c]** divide-by-constant (→ load-time reciprocal/shift), **[√]** sqrt,
**[Σ]** order-dependent accumulation, **[cmp]** float comparison/threshold, **[×64]** needs int64 intermediate.

### 2.1 WAVE — `wave_substep` (`atmosphere_solver.cpp:44-160`), runs `n_wave`× at the wave CFL

| Op | Site | Arithmetic (quoted) | Flag | Idiom |
|---|---|---|---|---|
| source feed | `:62-67` | `feed = wave_source[i]*feed_rate*dt; feed=min(feed,wave_source[i]); feed=min(feed,max_source_per_step); wave_p[i]+=feed; wave_source[i]-=feed` | [cmp] (`>0.001f`) | `mul_q16`; thresholds → Q16.16 compares |
| **wave Laplacian** | `:96-102` | `lap_i += std::min(perm_i, permeability[n]) * (wave_p[n] - p)` (4 faces) | [Σ] [×64] | **face-flux gather:** `min(perm)` is exact; `w·(p_n−p)` per face in int64; sum 4 → narrow. NOTE: this is wave (not conserved per-se — it's a relaxation operator on a zero-mean field), but the symmetric `(p_n−p)` form makes it the same once-per-face shape |
| **velocity kick** | `:108` | `wave_v[i] += (c_sq * lap[i] - damping * wave_v[i]) * dt` | [×64] **OVERFLOW** | `c_sq·lap` in int64, `−damping·wave_v`, **×dt before narrow** (plan M5). The headline watch-point |
| pressure update | `:113` | `wave_p[i] += wave_v[i] * dt` | [×64] | `mul_q16(wave_v, dt)` |
| per-cell absorb | `:124-131` | `a = wave_absorb[i]*absorb_strength*dt; k = (a<1)?(1-a):0; wave_v*=k; wave_p*=k` | [cmp] | `mul_q16`; `a<1` → Q16.16 compare; `scale_mag` (signed, shrink-only) for the `×k` since wave_v/wave_p are signed |
| wave BCs | `:134-139` | `if (is_wall||is_vacuum||obstacles) { wave_p=0; wave_v=0; }` | [cmp] bool mask | exact (integer 0) |
| **mean_wp reduction** | `:143-151` | `sum += wave_p[i]` over `!obstacle&&!wall&&!vacuum`; `mean_wp = (count>0)? sum/count : 0` | **[Σ] [÷] DEEPEST HAZARD** | int64 sum (order-free); **rounded** integer mean (§4, plan §4.2) |
| anomaly transfer | `:154-159` | `xfer = transfer*dt; atmosphere[i] += (wave_p[i] - mean_wp) * xfer` (interior mask) | [×64] [cmp] | `mul_q16`. **This is the write into the conserved atmosphere** — the mean exists to keep it mass-neutral (→ §3, §4.5) |

### 2.2 ATMOSPHERE DIFFUSION + BC + WIND — `diffuse_solve` (`:168-426`), runs ×1 at full `sim_time`

| Op | Site | Arithmetic (quoted) | Flag | Idiom |
|---|---|---|---|---|
| mu | `:183` | `const float mu = d_atm * dt` (`d_atm=200`, `dt≈0.0417` → `mu≈8.33`) | [cmp] (`mu>1e-8`) | scalar precompute per tick (dt floats per tick — see §5 cliffs) |
| rhs copy | `:197` | `rhs[i] = atmosphere[i]` | — | exact copy |
| **GS face weights** | `:241-255` | `face = (obstacle\|\|wall)?0:min(perm_i,perm[nb]); nb = Σ w_face*atmosphere[nb]; wsum = Σ w_face` | [Σ] | exact `min`; `w∈{0,½,1}→shift` collapse (plan M7) folds the `mul·w` into a shift |
| **GS per-cell divide** | **`:257`** | **`atmosphere[i] = (rhs[i] + mu*nb) / (1.0f + mu*wsum)`** | **[÷] [×64] S2c CENTERPIECE** | **precomputed reciprocal `Dinv[i]`** + **residual/flux form** (plan §3.1-3.2). NOT a per-cell int64 divide (unshippable on GPU). See §4 |
| GS-residual hook | `:274-301` | `res = (1+mu*wsum)*atm[i] - mu*nb - rhs[i]; res_max=max(...,|res|); ... res/atm_absmax` | [÷] (one scalar `/atm_absmax`) [cmp] | **read-only diagnostic** (`last_gs_residual`); nothing reads it in-sim. Keep as float CPU oracle OR mirror in integer for the convergence-acceptance test (plan S2c) |
| vac_dist BFS | `:322-355` | two passes, `vac_dist[i]=1/2` if a neighbour is `0/1` | bool/int only | **already integer** (uint8 BFS) — no float |
| eta + sponge BCs | `:357-380` | `eta = min(breach_rate*dt, 1.0); atmosphere[i] *= (1-eta)` / `(1-eta*0.5)` / `(1-eta*0.25)`; `wave_v *= (1-min(30*dt,1))` etc. | [×64] [cmp] | `mul_q16`; the `*0.5`/`*0.25`/`*0.5` are `>>1`/`>>2`/`>>1`. **atmosphere here is a relaxation/sink, not a flux — sheds mass to vacuum by design (NOT R2-CONS)** |
| **wind gradient** | `:413-420` | `p_side = p_here + f*(total(n)-p_here)` (4 sides); `wind_x = -(p_right-p_left)*0.5; wind_y = -(p_down-p_up)*0.5` (`total = atm+wave_p`) | [×64] | `mul_q16` for the face blend; `*0.5` → `shr_round0` (wind is signed). 2-term central diff |

### 2.3 SMOKE / GAS — `SmokeDynamics::step` + `backtrace_sample` (`smoke_dynamics.cpp:1-219`), runs `n_smoke`× per plane

| Op | Site | Arithmetic (quoted) | Flag | Idiom |
|---|---|---|---|---|
| neighbor blend | `:18-19` | `face = min(perm[self],perm[ni]); return f[self] + face*(f[ni]-f[self])` | [×64] | exact `min`; `mul_q16` |
| **smoke Laplacian** | `:163` | `lap[i] = s_up+s_down+s_left+s_right - 4.0f*s` | [Σ] | 5-term; **R2-CONS edge-flux** (conserved field) — gather faces, ± to both cells |
| **wind-dependent diffusion** | `:168-170` | `wind_sq = wind_x[i]²+wind_y[i]²; d_eff = d_smoke*(1 + wind_diffusion_scale*wind_sq); smoke[i] += d_eff*actual_dt*lap[i]` | [×64] | `mul_q16` chain. `wind_sq` is a square (int64), `wind_diffusion_scale=50` |
| dt_adv | `:183` | `dt_adv = advection_rate * actual_dt` (`advection_rate=900`) | [×64] | scalar `mul_q16` |
| **back-trace dist** | **`:63-64`** | **`dist = std::sqrt(bx*bx + by*by); steps = (int)std::ceil(dist)`** | **[√] [cmp]** | **fixed-iter integer sqrt** (FixPointCS-style; reuse S1's planned int-sqrt). `steps` is an **integer cliff** off the sqrt (a sub-step *count*) |
| **march inverse** | **`:66`** | **`inv = 1.0f / static_cast<float>(steps)`** | **[÷]** | `steps` is small int → `recip_mul` or a tiny reciprocal LUT keyed on `steps` |
| step bilinear-floor | `:72-73` | `ti = (int)std::floor(nxp + 0.5f); tj = (int)std::floor(nyp + 0.5f)` | [cmp] **round-half-up** | round-half-up integer idiom on Q16.16 (one of THREE back-trace rounding modes, plan §5.2) |
| sample floor | `:95-99` | `x0 = (int)std::floor(px); ... fx = px - (float)x0` | **floor toward −∞** | integer floor + fractional-part extraction in Q16.16 (second rounding mode) |
| bilinear weights | `:103-108` | `cw[k] = (1-fx)(1-fy)`, `fx(1-fy)`, `(1-fx)fy`, `fx·fy` | [×64] | `mul_q16` (fx,fy ∈ [0,1)) — **integer bilinear** (NEW helper, §8) |
| **bilinear renorm divide** | **`:122`** | **`return (wsum > 1e-6f) ? (acc / wsum) : src[y*w+x]`** | **[÷] M8 (SECOND dynamic divide)** | int64 `(acc<<16)/wsum`, small-divisor guard, OR fixed-weight LUT (third rounding mode). `acc = Σ cw[k]*src`, `wsum = Σ cw[k]` over non-sealed corners |
| clamp | `:213,272` | `smoke[i] = std::clamp(smoke[i], 0.0f, 1.0f)` | [cmp] | integer clamp to `[0, FP_ONE]` |
| **sink_hop** | `:225-277` | same `backtrace_sample`; `bx = min(sink_strength,1.0)*sink_x[i]` | [cmp] | same idioms; the displacement is the sink direction (capped 1 cell) |

### 2.4 ORCHESTRATION — `run_substeps` (`physics_engine.cpp:111-252`)

| Op | Site | Arithmetic (quoted) | Flag | Idiom |
|---|---|---|---|---|
| **n (wave) cliff** | **`:125-126`** | **`dt = (double)atmos.max_dt(); n = max(1, (int)std::ceil((double)sim_time/dt))`** (`max_dt = 0.5/c`, `:7-10`) | **[÷] [√-free] CLIFF** | `max_dt` is config-constant → precompute Q16.16; `ceil_div` (S1 has it, `fixed_point.h:189`) |
| dt_actual | `:129` | `dt_actual = (double)sim_time / n` | [÷] | passed as float to wave_substep; Q16.16 `recip_mul` or div-by-int-n |
| **max_wind_sq reduction** | **`:183-187`** | **`ws = wind_x[i]²+wind_y[i]²; if (ws>max_wind_sq) max_wind_sq=ws`** | **[Σ-max] [cmp]** | integer max over the Q16.16 wind field (order-free); the square needs int64 |
| d_smoke_max reduction | `:188-191` | `if (gas_diffusion[gi] > d_smoke_max) d_smoke_max = ...` | [cmp] max | integer max (constant per config) |
| **n_smoke cliff** | **`:192-198`** | **`d_eff_max = d_smoke_max*(1 + wind_diffusion_scale*max_wind_sq); dt_stable = 1/(4*d_eff_max); n_smoke = max(1,(int)std::ceil(sim_time/dt_stable))`** | **[÷] [×64] SPATIAL-MAX CLIFF** | the spatial-max d_eff cliff — compute `d_eff_max` integer, `dt_stable` reciprocal, `ceil_div`. Depends on a *runtime* reduction (max_wind_sq) → the hard one |
| dt_smoke | `:203` | `dt_smoke = (float)((double)sim_time / n_smoke)` | [÷] | div-by-int |
| **`.any()` per plane** | `:212-216, 238-242` | `for i: if (gas_slice[i] != 0.0f) { any=true; break; }` | [cmp] | exact integer `!= 0` (must compare the integer field, not a float bridge) |
| K sink loop | `:234-251` | `K = smoke.vent_hops` (=16) | const | integer loop count from config |

---

## 3. Conservation — which fields, and which idiom

| Field | Conserved? | Why / where | Idiom |
|---|---|---|---|
| **atmosphere** | **YES (bulk)** | `mean_wp` exists *only* to keep the wave→atmosphere transfer mass-neutral (`:154-159`); the GS diffusion `(I−μΔ)` is a redistribution; total air in a sealed region must hold (P2 test). | **R2-CONS edge-flux** for the GS stencil + transfer. **EXCEPTION:** the vacuum/sponge BC (`:357-380`) is a deliberate **sink** (air vents to space) — NOT conserved there, that's correct (like temperature cooling). |
| **wave_p** | mass-neutral (zero-mean) | not a mass field; `mean_wp` subtract makes the transfer DC-free | symmetric face form (`:96-102`); the conservation concern is the *transfer's* DC bias (§4.2 plan: rounded mean) |
| **wave_v / wave_source** | no | velocity / injected energy | plain `mul_q16` + `scale_mag` for signed scaling |
| **wind_x / wind_y** | no (derived) | `wind = −∇(atm+wave_p)` each tick — recomputed, not accumulated | plain gradient; no conservation |
| **gas / smoke (5 planes)** | **YES (each plane)** | `[0,1]` density advected + diffused; total smoke must hold over a settle (P2). Advection is the **semi-Lagrangian back-trace** (NOT a flux-divergence form like water) — conservation here is via the **bilinear-renorm** (the `acc/wsum` divide re-normalizes the gathered weights so it neither creates nor destroys where corners are excluded). | **R2-CONS** for the *diffusion* Laplacian (`:163`); the *advection* back-trace is **semi-Lagrangian (not edge-flux)** — its mass-property comes from the renorm divide, NOT from a gather-once-±-pair. This is the one place the S1 conservative-flux pattern **does NOT directly apply** (see §7, the back-trace hazard). |

**Where the S1 conservative-flux pattern (gather-once / ±flux / narrow-shared, `fixed_point.h:85-98`) DOES apply:**
- the **wave Laplacian** faces (`:96-102`) — symmetric `(p_n−p)`, gather once, apply ±.
- the **atmosphere GS stencil** (`:251-257`) in residual/flux form — each face flux once, ± to both cells.
- the **smoke diffusion Laplacian** (`:163`) — each face flux once, ± to both cells.

**Where it does NOT apply (relaxation / sink / semi-Lagrangian):**
- the **vacuum/sponge BC** (`:357-380`) — a one-sided decay (sink to space), use plain `mul_q16`/shift.
- the **smoke/gas advection back-trace** (`:52-123`) — semi-Lagrangian; conservation via renorm divide, not flux pairs. (P2 test must watch this hardest.)
- the **wind gradient** (`:413-420`) — a derived field, no conservation.

---

## 4. The Red-Black GS structure (the S2c centerpiece)

**Confirmed 2-color schedule** (`:201-205`): `for color in {0,1}: ... if (((x+y)&1) != color) continue;`.
Determinism property holds **by construction** — red cells (`(x+y)&1==0`) read only black neighbours and
vice versa; there is **no intra-color read-after-write**, so the result is identical on any architecture
(plan §3.3 #4: a *schedule to pin*, not a desync). On GPU this is two kernel launches with a barrier.

**Operator:** `(I − μΔ) atm_new = rhs` with `rhs = atmosphere` (the IMEX `u*`), `μ = d_atm·dt ≈ 8.33`
(`d_atm=200`). Per cell: `(1 + μ·wsum)·atm[i] − μ·Σ w_face·atm[nb] = rhs[i]`. Run `gs_iters = 8` iterations
× 2 colors per tick (×1 tick — diffuse runs **once** per tick, not per wave-substep; the plan's "stale 48"
note confirms this is `physics_engine.cpp:165`, the single `diffuse_solve` call).

**The divide** (`:257`): `atmosphere[i] = (rhs[i] + mu*nb) / (1.0f + mu*wsum)`. This is **one of three**
genuine dynamic divides in the group (the others: `mean_wp sum/count`, the smoke bilinear `acc/wsum`).

**How the precomputed-reciprocal `Dinv` multiply replaces it (plan §3.1-3.2):**
1. Per tick, precompute `Dinv[i] = reciprocal_q16(ONE_Q16 + mul(mu, wsum_i))` ONCE (the one slow op, done
   on changed cells only). Rebuild trigger keyed on `(mu | obstacles | is_wall | is_vacuum | permeability)`.
   Explicit `continue` on skipped cells, **never `Dinv=0`**.
2. The hot sweep is a **multiply in residual/flux form** (NOT the quotient form — the quotient form has no
   fixed point under a truncating multiply at `μ·wsum≈33` → systematic leak, plan §3.2):
   `atm[i] += mul( Σ_face mul(mul(mu, w_face), (atm[n]-atm[i])) − (atm[i]-rhs[i]), Dinv[i] )`.
   Equal neighbours at the fixed point → increment truncates to exactly 0 → drift-free.
3. The `w∈{0,½,1}→shift` collapse (plan M7) folds the `mul(mu,w_face)` 3-factor product into a shift since
   permeability is `{0,0.5,1.0}` today (collapses `mul·w·diff`).

**Acceptance** (plan S2c): self-reproduction at `tol=0.0` AND the integer GS Linf residual (the `:274-301`
hook, mirrored in integer) within a stated factor of the float build's on the stress scenarios.
"Deterministic" and "converges" are separate claims — both tested.

---

## 5. Integer cliffs — float-derived substep counts to make fixed-point

| Cliff | Site | Formula | Inputs | Fix |
|---|---|---|---|---|
| **n_wave** (= n) | `physics_engine.cpp:125-126` | `n = max(1, ceil(sim_time / max_dt))`, `max_dt = 0.5/c` (`atmosphere_solver.cpp:9`) | **config-constant** (`c=66`) — `max_dt` is fixed | Precompute `max_dt` as Q16.16 at load; `ceil_div(sim_time_q, max_dt_q)` — `fixed_point.h:189` already has `ceil_div`. The easy cliff (no runtime reduction). |
| **n_smoke** | `physics_engine.cpp:192-198` | `n_smoke = max(1, ceil(sim_time / dt_stable))`, `dt_stable = 1/(4·d_eff_max)`, `d_eff_max = d_smoke_max·(1 + wind_diffusion_scale·max_wind_sq)` | **RUNTIME** (`max_wind_sq` is a spatial-max reduction over the live wind field) | The HARD cliff: `max_wind_sq` = integer max over Q16.16 wind (order-free, `:183-187`); square in int64; `d_eff_max`, `dt_stable` reciprocal, `ceil_div`. A 1-ULP slip flips the *substep count* → peers run different iteration counts → total desync (plan §4.4 rank-2 hazard). |
| **steps** (back-trace) | `smoke_dynamics.cpp:64` | `steps = ceil(sqrt(bx²+by²))` | per-cell `bx,by` (= wind·dt_adv) | fixed-iter integer sqrt → `ceil`. A per-cell march-count cliff (smaller blast radius than n_smoke, but still a count derived from a transcendental). |

Note: water's `n` cliff is **already done** in S1 (`physics_engine.cpp:292`, `ceil_div`). S2's three cliffs
reuse the same `ceil_div` helper; the **new** burden is `max_wind_sq` (a runtime reduction feeding a cliff).

---

## 6. Intra-group couplings — the reason they migrate TOGETHER

```
                wave_source ──► wave_p ──► [mean_wp Σ] ──► atmosphere(transfer)
                                  │                              │
                                  │                              ▼
                                  │                    [RB-GS diffusion μΔ]  (S2c)
                                  ▼                              │
                          wave_p (for wind)                      ▼
                                  └──────────► wind = −∇(atm + wave_p)  (S2c)
                                                       │
                                                       ▼
                              wind ──► smoke/gas advection + d_eff diffusion  (S2b)
                                       │                    ▲
                                       │              max_wind_sq cliff (n_smoke)
                                       ▼
                                  sink_hop (K×)
```

**The float bridges that close only when the whole group is integer:**
- **mean_wp → atmosphere** (`:157`): wave_p (S2a) writes the conserved atmosphere (S2c). Same group.
- **atmosphere + wave_p → wind** (`:395, 413-420`, S2c): both inputs in-group.
- **wind → smoke advection** (`smoke_dynamics.cpp:200-201`, S2b consumes S2c's wind): the producer (wind, S2c)
  and consumer (smoke, S2b) are in the same group — **wind is produced AFTER smoke is listed (S2b) in plan
  order, but within one tick wind is computed in `diffuse_solve` BEFORE the smoke loop** (`physics_engine.cpp:165`
  then `:208`). So S2b's smoke reads S2c's wind: the group must land atomically, even though the sub-steps
  are authored in S2a→S2b→S2c order.
- **wind → n_smoke cliff** (`:183-198`): the substep count itself depends on the integer wind.
- **(diffusion → atmosphere)**: the GS redistributes the same atmosphere the transfer wrote.

**Sequencing for the S2 plan** (plan §6.3): **S2a** = wave + mean_wp (int64 rounded mean + `c_sq·lap`
dt-order discipline); **S2b** = smoke + 5 gas planes (R2-CONS diffusion, semi-Lagrangian back-trace with
fixed-iter sqrt + the 3 rounding modes + the bilinear renorm divide M8) + `n_smoke` cliff; **S2c** =
atmosphere diffusion (RB-GS residual/flux + Dinv) + wind. Because wind (S2c) feeds smoke (S2b) within a
tick, the three sub-steps are gated together — each must self-match at `tol=0.0`, and the group is only
*cross-GPU-deterministic* once all three are integer (the within-group float bridges are gone).

**Cross-group bridges OUT of S2 (stay float until their system migrates):**
- **water W3/W5 → atmosphere/gas** (`physics_engine.cpp:305-371`, S1 already dequantizes at this boundary) —
  becomes integer-clean when S2 lands (water is already integer; the bridge is on the atmosphere/gas side).
- **fire → atmosphere/smoke/wind reads** (S3, later) — fire reads wind/atmosphere; bridge until S3.

---

## 7. Determinism hazards (ranked)

1. **`mean_wp` global reduction** (`:143-151`) — **#1, the deepest.** A float sum's parenthesization differs
   across CPU-scalar / SIMD / CUDA-warp; subtracted from *every* cell → contaminates the whole atmosphere
   field that tick. **Integer fix is order-free for free** (int64 sum, plan §4.2), BUT two sharp edges:
   (a) `sum` is *already Q16.16* → `sum/count` has **no `<<16` pre-shift** (differs from the GS divide which
   does pre-shift — M3); (b) signed truncation toward zero biases the mean with `sign(sum)` → a DC drift
   into every cell → a **P2 conservation defect** → use **round-to-nearest-even** on the mean (plan §4.2).
2. **smoke/gas advection back-trace float interpolation** (`smoke_dynamics.cpp:52-123`) — the bilinear
   sample (`acc/wsum`, `:122`), the `sqrt`-derived step count (`:64`), the two floor modes (`:72-73`, `:95`),
   and the `1/steps` (`:66`) are all float today. Semi-Lagrangian → NOT conservative by a flux pair; mass
   property rides on the renorm divide. **This is the hardest field to keep both deterministic (P1) AND
   conserving (P2).** Three distinct rounding modes must each map to a pinned integer idiom (plan §5.2).
3. **n_smoke cliff over `max_wind_sq`** (`:183-198`) — a 1-ULP float slip flips an **integer substep count**
   → peers iterate a different number of times → total desync; a naive within-substep digest misleads
   (plan §4.4). The reduction `max_wind_sq` is a max (order-free for integers) but **must read the integer
   wind**, and the cliff arithmetic must be fixed-point.
4. **GS red-black read-after-write** (`:199-261`) — **NOT a cross-arch hazard** once the fixed two-color
   schedule is pinned (red reads only black, no intra-color RAW). Confirmed order-independent (plan §3.3 #4).
   The hazard is only the *divide* (→ Dinv) and the *quotient-vs-residual* form choice (§4).
5. **Signed values** — `wave_p`, `wave_v`, `wind_x/y` are signed → arithmetic right-shift rounds toward −∞;
   use `shr_round0` / `scale_mag` (`fixed_point.h:160,175`) where symmetric magnitude behaviour is wanted
   (the `*0.5` wind gradient, the `*k` wave absorb). `mul_q16`'s toward-−∞ truncation is fine for the
   conservative face pairs (it cancels ± exactly).
6. **Other order-dependent sums** — `max_wind_sq` (max, order-free), `d_smoke_max` (max), the per-plane
   `.any()` (OR, order-free): all order-free for integers, BUT each membership/`!=0` predicate **must be
   evaluated on the integer field**, never a float bridge (plan §4.6).
7. **The float comparisons / thresholds** — `wave_source>0.001` (`:62`), `a<1.0` (`:127`), `mu>1e-8`
   (`:192`), `boil_p_thresh` reads, `gas!=0` (`:213`): each becomes a Q16.16 integer compare; pin the
   threshold constants in Q16.16.

---

## 8. Reuse from S1 (`fixed_point.h`) + NEW helpers S2 needs

**Reuse directly (already in `fixed_point.h`):**
- `quantize` / `dequantize` / `dequantize_f` (`:59-74`) — the boundary casts (render + cross-group bridges).
- `mul_q16` (`:81`) + `mul_wide` (`:89`) + `narrow` (`:96`) — every Q16.16 multiply + the conservative
  flux gather (gather wide, ± to both cells, narrow once) for the wave/GS/smoke Laplacians.
- `make_recip` / `recip_mul` (`:121-154`) — the constant divides (`dt_actual=sim_time/n`, `dt_stable`,
  `1/steps`); also the building block for the GS `Dinv` and the mean.
- `shr_round0` (`:160`) — symmetric decay for signed fields (the `*0.5`/`*0.25` sponge, wind `*0.5`).
- `scale_mag` (`:175`) — shrink-only signed scale (the wave absorb `×k`, k∈[0,1]).
- `ceil_div` (`:189`) — **all three S2 cliffs** (n_wave, n_smoke, back-trace steps) reuse it.
- (`tan_poly` `:216` — not needed by S2; water-only.)

**NEW helpers S2 needs (to add to `fixed_point.h` or a sibling):**
1. **Deterministic integer reduction for `mean_wp`** — `int64 sum_over_mask` + a **rounded** integer mean
   `(sum + round_bias)/count` (round-to-nearest-even, sign-symmetric, pinned). The single most important new
   primitive; also the Spike-0 artifact. NO `<<16` pre-shift (sum is already Q16.16, M3).
2. **Q16.16 reciprocal `reciprocal_q16(denom)` for the GS `Dinv`** — a per-cell reciprocal of
   `(ONE_Q16 + mul(mu, wsum))`, computed once per tick per changed cell (the one slow op). Likely a
   reciprocal table/Newton refine; `make_recip` is double-at-load and won't serve a *runtime* per-cell
   divisor — this needs an integer-only reciprocal (fixed-iter, GPU-portable).
3. **Fixed-iteration integer `sqrt`** — for the back-trace step count (`:64`). The plan flags S1/S2/S3 all
   want it; if S1 lands it (its `max_dt` sqrt, S1 Q-S1-3), S2 reuses it. Must be the SAME routine on every
   peer (not host `std::sqrt`).
4. **Integer bilinear sample** — the back-trace's 4-corner `cw[k]` weights (`:103-108`) + the renorm
   `acc/wsum` divide (`:122`) as a single deterministic integer routine (the M8 second dynamic divide):
   int64 `(acc<<16)/wsum` with a small-divisor guard, or a fixed-weight LUT. Carries the three rounding
   modes (floor−∞, round-half-up, the renorm) explicitly.
5. **Integer max-reduction over a Q16.16 field** — `max_wind_sq` (order-free; the square needs int64).
   Trivial but worth a named helper since it feeds a cliff.

---

## 9. Quick cross-reference — the canonical hazard sites (for the S2 plan)

| Plan reference | This map | Site |
|---|---|---|
| mean_wp (#1 reduction) | §2.1, §4, §7.1, §8.1 | `atmosphere_solver.cpp:143-151` |
| GS per-cell divide (S2c centerpiece) | §2.2, §4 | `atmosphere_solver.cpp:257` |
| wave_v `c_sq·lap` overflow (M5) | §1, §2.1 | `atmosphere_solver.cpp:108` |
| RB-GS 2-color schedule | §4, §7.4 | `atmosphere_solver.cpp:201-205` |
| wind = −∇(atm+wave_p) | §2.2, §6 | `atmosphere_solver.cpp:413-420` |
| back-trace sqrt + 3 rounding modes + renorm divide (M8) | §2.3, §3, §7.2, §8.4 | `smoke_dynamics.cpp:64,72-73,95,122` |
| n_wave cliff | §5 | `physics_engine.cpp:125-126` |
| n_smoke cliff over max_wind_sq | §5, §7.3 | `physics_engine.cpp:183-198` |
| per-gas `.any()` / per-plane loop | §2.4 | `physics_engine.cpp:208-226, 234-251` |

*All line numbers verified at HEAD (2026-06-24). Re-verify before impl — Patch 1/2 moved this code and the
S2 work will move it again.*
