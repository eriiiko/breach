# S2 — Atmosphere / Wave / Wind / Smoke / Gas → integer (Q16.16) — plan

**Status:** plan, awaiting Erik's OK (the §5 decisions, headlined by the smoke-advection switch).
The **second field migration** of the fixed-point arc and **the last big coupling group before fire (S3)**.
Built from `docs/s2_atmosphere_group_map.md` (the factual HEAD survey — every divide, transcendental,
reduction, conservation property and cliff), `docs/s2_advection_research.md` (the smoke-advection
recommendation), `docs/fixed_point_migration_plan.md` §9 (the locked resolutions), and the shipped S1
toolkit `cpp/src/fixed_point.h` + `docs/s1_water_fixed_point_plan.md` (the conservative gather-once / ±-pair /
shared-narrow template).

**Unlike S1, this group is HARD:** it has a global reduction (`mean_wp`), three genuine dynamic divides
(`mean_wp`, the GS per-cell divisor, the smoke bilinear renorm), a per-cell transcendental (the back-trace
`sqrt`), two runtime-derived integer cliffs (`n_wave`, `n_smoke`), and **five systems that couple inside one
tick** so they must migrate **atomically**. The plan's headline move is to *delete* most of that hazard
surface by **switching smoke/gas from semi-Lagrangian back-trace to conservative flux-form** (research §9).

All HEAD line numbers re-verified 2026-06-24 against `atmosphere_solver.cpp` (`:108`, `:143-151`, `:154-159`,
`:240-257`), `physics_engine.cpp` (`:183-198`), `smoke_dynamics.cpp` (`:52-277`). Re-verify before impl —
Patch 1/2 moved this code and S2 will move it again.

---

## 1. Scope — what converts, what doesn't, and why it lands atomically

### 1.1 The five systems migrate as ONE group

The per-tick DAG is a producer→consumer chain (`map` §6) and the within-group float bridges only close when
**all** of it is integer:

```
wave (explicit, n_wave×) → mean_wp (global Σ) → atmosphere transfer
   → atmosphere diffusion (implicit RB-GS, ×1) → wind = −∇(atm+wave_p)
   → smoke/gas advection + diffusion (n_smoke× on the once-computed wind) → sink venting (K×)
```

Critically, **wind is produced in S2c (`diffuse_solve`) but consumed in S2b (the smoke loop)** within the
same tick (`physics_engine.cpp:165` then `:208`). So even though the sub-steps are *authored* S2a→S2b→S2c,
the group is only *cross-GPU-deterministic* once all three are integer. We gate each sub-step's P1/P2
internally (via a temporary float-bridge ordering, §6.2), but they **land together** in one merge.

### 1.2 CONVERT to int32 Q16.16 (the synced state)

| Field | Shape today | Signed | Conserved | Q16.16 sizing | GPU width (frozen now) |
|---|---|---|---|---|---|
| **atmosphere** (pressure) | (h,w) f32 | no | **YES (bulk)** | ~1–2 interior, comfortable in ±32768; smallest meaningful increment `xfer·(wave_p−mean_wp)`~1e-3 → ~65 counts | **int32** |
| **wave_p** | (h,w) f32 | **yes** | mass-neutral (zero-mean) | comfortable | int32 |
| **wave_v** | (h,w) f32 | **yes** | no | **OVERFLOW WATCH** — `c_sq·lap` exceeds ±32768 *before* ·dt (§3, S2a) | int32, **Q24.8 exception candidate** |
| **wave_source** | (h,w) f32 | no | no | comfortable | int32 |
| **wind_x / wind_y** | (h,w) f32 each | **yes** | no (derived) | 2-term central diff ×0.5; feeds smoke + the `max_wind_sq` cliff → must share smoke's read format | int32 |
| **gas** — 5 planes (white_smoke, **black_smoke**, poison, teargas, fuel_gas) | (5,h,w) f32 | no | **YES (each plane)** | `[0,1]`-clamped tracers; Q16.16 res ≈1.5e-5 ≪ perceptual; no overflow in the flux `mul_wide` | **int16 (Q1.15) frozen, ship int32 on CPU** |
| **smoke** | **view of gas[BLACK_SMOKE]** | no | YES | *same storage as a gas plane* — black_smoke IS plane 1 | int16 (Q1.15) |

`smoke` and "the 5 gas planes" are the **same storage** (`gamemap.py:109`); migrating gas migrates smoke.

### 1.3 The GPU int16 (Q1.15) candidates — frozen now, int32 on CPU

The §9-locked decision: the **`[0,1]`-clamped fields (all 5 gas planes incl. smoke)** are the int16(Q1.15)
bandwidth win on GPU (5 planes = the widest sim memory). The physical fields
(atmosphere/wave_p/wave_v/wave_source/wind — signed and/or >1 range) stay **int32**. **Ship int32 on CPU
now**; **record int16(Q1.15) for the gas planes in the format-version tag this session** so the CUDA buffers
+ digest schema are designed once. int16 add/mul stay associative/exact → zero determinism cost; the width
freeze is a forward commitment, not a CPU change. (Open question Q-S2-6.)

### 1.4 STAYS FLOAT — the cross-group bridges OUT of S2 (until their system migrates)

- **water W3 / W5 → atmosphere/gas** (`physics_engine.cpp:305-371`): water is already int (S1); these
  bridges read/write atmosphere + gas, which become integer when S2 lands → the bridge **closes on the
  S2 side** here (the atmosphere/gas dequantize at this boundary goes away). S1 already dequantizes water
  at this boundary; after S2, the boundary is integer↔integer (no float).
- **fire → atmosphere / smoke / wind reads** (S3, later): fire reads wind/atmosphere/smoke and writes heat
  sinks; **this is the ONLY float bridge S2 leaves open** (downstream, to fire). S2 dequantizes for fire at
  that boundary until S3. Per the brief: *no float-bridge-to-a-later-group except fire downstream.*
- **render**: dequantize on demand at the renderer boundary (one source of truth = the int field).

After S2 lands, the entire atmosphere/wave/wind/smoke/gas group + water is cross-GPU-deterministic; only the
fire coupling remains float until S3.

---

## 2. The Q16.16 idioms reused + the NEW helpers

### 2.1 Reused verbatim from `fixed_point.h` (S1's shipped toolkit)

- `quantize` / `dequantize` / `dequantize_f` (`:59-74`) — the load/render/cross-group boundary casts.
- `mul_q16` (`:81`) + `mul_wide` (`:89`) + `narrow` (`:96`) — every Q16.16 multiply **and the conservative
  flux gather** (gather wide, ± the same int64 to both cells, narrow once) for the wave Laplacian, the
  atmosphere GS stencil, the smoke diffusion Laplacian, and the **new flux-form smoke advection**.
- `make_recip` / `recip_mul` (`:121-154`) — the *loop-invariant* divides (`dt_actual = sim_time/n`,
  `dt_stable`, `mu = d_atm·dt`). **NOT** valid for per-cell runtime divisors (the GS `Dinv`, the renorm) —
  `make_recip` is double-at-load for a single divisor.
- `shr_round0` (`:160`) — symmetric decay for signed fields (the sponge `*0.5`/`*0.25`, the wind `*0.5`).
- `scale_mag` (`:175`) — shrink-only signed scale: the wave absorb `×k` (k∈[0,1]), **and the smoke
  monotone outflow limiter** (the bounded `smoke[i]/out_sum` clamp, reused from water verbatim).
- `ceil_div` (`:189`) — **all three S2 cliffs** (n_wave, n_smoke, back-trace step-count if any survives).
- (`tan_poly` `:216` — water-only, not needed by S2.)

### 2.2 NEW helpers S2 needs

1. **Deterministic integer mean reduction** (`mean_wp`, the Spike-0 artifact, §3 S2a) — `int64
   sum_over_mask` (order-free) then a **rounded** integer mean. Two sharp edges from `map` §7.1:
   (a) `sum` is **already Q16.16** → `sum/count` has **NO `<<16` pre-shift** (differs from the GS divide
   which *does* pre-shift); (b) signed truncation toward 0 biases the mean by `sign(sum)` → a DC drift into
   every cell → a P2 defect → use **round-to-nearest-even** on `(sum + round_bias)/count`, sign-symmetric,
   pinned identically on every peer. This is the single most important new primitive.

2. **Integer max reduction over a Q16.16 field** (`max_wind_sq`, §3 S2b cliff) — `max` is **order-free for
   integers** (trivially, unlike a sum); the per-cell square `wind_x²+wind_y²` needs int64. A named helper
   because it feeds the n_smoke cliff. (`d_smoke_max` is also a max but config-constant.)

3. **Per-cell GS reciprocal `Dinv` for the RB-GS divisor** (§3 S2c, the centerpiece) — the divisor
   `1 + μ·wsum` is **per-cell AND continuous** (Q4 kept permeability continuous, so `wsum` is not just
   `{0,½,1}` sums — a per-cell reciprocal is genuinely needed each tick). `make_recip` cannot serve it
   (it's double-at-load for one divisor). **This is an OPEN DESIGN CHOICE (Q-S2-3):**
   - **(a) `/fp:strict` double reciprocal-then-quantize** — `Dinv[i] = quantize(1.0 / (1+μ·wsum_real))`,
     correctly-rounded double divide → deterministic *for a given divisor* (IEEE div is correctly rounded
     cross-vendor), then quantized. **Pro:** simplest, provably correct-rounded, cheap. **Con:** keeps a
     float op in the precompute → *not GPU-pure*, and the quantize step must be bit-identical to the GPU's
     eventual integer path or the two builds diverge. Acceptable on CPU; a liability for the CUDA port.
   - **(b) fixed-iteration integer reciprocal** (Newton / FixPointCS normalize+LUT+refine) — pure integer,
     GPU-clean, same routine on every peer. **Pro:** the honest endpoint; no float in the hot path; reused
     by any future per-cell divide. **Con:** more code; must be validated to a stated reciprocal precision.
   - **Recommendation:** ship **(a) on CPU now** (with the quantize pinned), but **design the helper's
     signature so (b) drops in for the GPU port** — i.e. treat `Dinv` as an opaque `reciprocal_q16(denom)`
     whose CPU body is the double path and whose GPU body is integer Newton. Flag the precompute float as a
     known CPU-only artifact in the format tag. **Pin this with Erik.**
   The `Dinv` is rebuilt **only on changed cells** each tick, keyed on `(mu | obstacles | is_wall |
   is_vacuum | permeability)`; skipped cells `continue` (never `Dinv=0`).

4. **Flux-form advection machinery (the divide-free limiter)** — pure integer `min`/`max`/`minmod` of two
   one-sided Q16.16 differences (research §3 Route (a)). **No new divide, no transcendental.** Reuses
   `mul_wide`/`narrow`/`scale_mag` for the flux gather + outflow clamp; the limited correction flux folds
   into the face flux **before** the shared narrow so it cannot break conservation.

### 2.3 What we explicitly do NOT need (a consequence of going flux-form)

Per research §5: **no integer `sqrt` and no integer bilinear** — both were *semi-Lagrangian-only*
(the back-trace march length `sqrt` at `smoke_dynamics.cpp:64` and the 4-corner bilinear `acc/wsum` at
`:122`). The flux switch **deletes** them, along with the three back-trace rounding modes and the `1/steps`
divide. (If Erik instead picks the keep-SL fallback in Q-S2-1, then integer sqrt + integer bilinear +
the renorm divide all come *back* into scope — a major reason to switch.)

---

## 3. Sub-steps (each its own gated commit on an `s2-atmosphere-fixedpoint` branch)

### S2a — wave + mean_wp

The explicit wave system (`wave_substep`, `atmosphere_solver.cpp:44-160`), run `n_wave`× per tick.

- **source feed** (`:62-67`): `mul_q16` for `feed_rate·dt`; the `min(feed, …)` clamps stay exact integer
  `min`; the `wave_source > 0.001f` threshold → a pinned Q16.16 compare constant.
- **wave Laplacian** (`:96-102`): the symmetric `min(perm)·(wave_p[n]−p)` face form is **already the
  gather-once shape** — `min(perm)` exact; `w·(p_n−p)` per face as `mul_wide`; sum 4 faces in int64; narrow
  once. (It's a relaxation operator on a zero-mean field, but the per-face shape is identical to S1's.)
- **velocity kick** (`:108`, **the headline overflow watch**): `wave_v[i] += (c_sq·lap − damping·wave_v)·dt`.
  With `wave_c=66 → c_sq=4356` and `lap` O(10), **`c_sq·lap` ≈ 4e4 exceeds ±32768 BEFORE the ·dt**. Fix:
  carry `c_sq·lap` and `−damping·wave_v` in **int64**, apply **`·dt` BEFORE narrowing** (the only safe
  order). **Measure peak |wave_v| in a blast scenario** (Spike): if even the post-dt `wave_v` exceeds the
  Q16.16 range, **wave_v takes a Q24.8 format exception** (8 fractional bits, ±8.4M integer range). This is
  a **format decision to pin with Erik (Q-S2-2)** — Q24.8 for wave_v ripples into the wave_p update (`:113`,
  `mul_q16(wave_v, dt)`) and the wind gradient's `wave_p` read. Recommendation: **measure first**; default
  to int32 Q16.16 with the dt-before-narrow discipline if peak |wave_v| stays inside ±32768, else Q24.8 for
  wave_v alone (wave_p stays Q16.16, the kick converts at the `·dt` narrow).
- **pressure update** (`:113`): `mul_q16(wave_v, dt)` (or the Q24.8 cross-format multiply if Q-S2-2 fires).
- **per-cell absorb** (`:124-131`): `mul_q16` for `a`; `a < 1` → Q16.16 compare; the `×k` (k∈[0,1], shrink)
  uses **`scale_mag`** since wave_v/wave_p are signed (shrink-only, magnitude-symmetric).
- **wave BCs** (`:134-139`): exact integer `0` on walls/vacuum/obstacles.
- **mean_wp** (`:143-151`, **the #1 determinism hazard**): `int64 sum` over the `!obstacle&&!wall&&!vacuum`
  mask (order-free); then the **NEW rounded integer mean** (§2.2 #1 — round-to-nearest-even, NO `<<16`
  pre-shift, sign-symmetric). **Stopgap now; plan the retirement:** §9-locked, `mean_wp` retires to a
  **local edge-flux** form (the transfer becomes a per-face flux that is DC-free by construction, killing
  the global reduction entirely). For S2a we ship the rounded-mean stopgap (correct + deterministic) and
  **scope the edge-flux retirement as a follow-up** (it changes the transfer's spatial structure → a
  feel-gated behaviour change, not bit-compatible with the mean form → defer to its own commit).
- **anomaly transfer** (`:154-159`): `mul_q16` for `xfer`; `atmosphere[i] += (wave_p[i]−mean_wp)·xfer` —
  this is the **write into the conserved atmosphere**; the rounded mean keeps it DC-free (P2-critical).
- **n_wave cliff** (`physics_engine.cpp:125-126`): `max_dt = 0.5/c` is **config-constant** → precompute as
  Q16.16 at load; `ceil_div(sim_time_q, max_dt_q)`. The **easy cliff** (no runtime reduction). `dt_actual =
  sim_time/n` via `recip_mul` or div-by-int-n.

**Gate S2a:** P1 self-match `tol=0.0`; P2 — the wave→atmosphere transfer is **mass-neutral to the LSB** (the
rounded mean is the test's whole point — a biased mean is a DC leak into atmosphere). No `c_sq·lap` overflow
in the blast stress scenario (assert the int64 intermediate + measured peak).

### S2b — smoke + 5 gas (the flux-form switch — the biggest behaviour change of the arc)

Per `s2_advection_research.md` §9: **SWITCH from semi-Lagrangian back-trace to conservative flux-form,
specifically donor-cell upwind + a divide-free two-slope MC/minmod limiter (Family 3c).** This is the
recommended default; Q-S2-1 confirms it vs the keep-SL fallback. Authored as research §6's gated sequence:

- **S2b-0 — representation:** quantize smoke + 5 gas planes to int32 Q16.16 (`[0,1]` tracers share the
  water/heat scale); float dequantize for the renderer + the fire bridge.
- **S2b-1 — bare donor-cell upwind flux (Family 3a):** port `water_solver.cpp`'s flux section verbatim —
  `v_face = (wind[i]+wind[n])>>1` (`shr_round0`, signed), `donor = upwind smoke`, `flux = mul_wide(v_face,
  donor)`, reuse water's **`flux_to_dq` shared-narrow** (the same lambda, **not re-derived** — the MSVC vs
  clang/gcc 128-bit narrow is already pinned by `tests/_s1_flux_truncation_check.py`), the **`scale_mag`
  outflow limiter** (`out_sum = Σ outgoing dq`; if `out_sum > smoke[i]` scale by `smoke[i]/out_sum` toward
  0 — the bounded exact integer divide, the conservation clamp), and the `±` apply. A solid face carries no
  flux (`!solid[i] && !solid[n]`) → **the wall-clip anti-tunnelling march disappears by construction.**
  Smoke is *easier* than water (no own velocity field, no surface potential). **Gate P1 (`tol=0.0`) + P2
  (sealed-room `Σ smoke` LSB-constant)** — the conservation+determinism milestone, independent of feel.
- **S2b-2 — the divide-free MC/minmod limited correction flux (Family 3c):** add the limited anti-diffusive
  face flux. The limiter is `min`/`max`/`minmod` of the two one-sided Q16.16 differences `(s_i−s_{i−1})` and
  `(s_{i+1}−s_i)` — **pure integer, never forms the ratio `r`, no divide** (research §3 Route (a)). The
  limited correction is **folded into the face flux BEFORE the shared narrow** → any limiter truncation
  still cancels in the `±` pair → cannot break conservation. **`scale_mag`** is the monotone outflow
  primitive for the correction. **Default limiter: MC** (smooth-smoke look); **minmod** the safe fallback
  (Q-S2-1b). **Gate P1+P2 + the feel A/B** (SSIM on rendered frames vs the float SL golden — still reads as
  crisp wisps).
- **S2b-3 — wind-dependent diffusion → R2-CONS edge-flux + the n_smoke cliff:** the Pass-A wind-coupled
  Laplacian (`smoke_dynamics.cpp:163-170`) migrates to the **edge-flux idiom** — `flux = mul(d_eff_face,
  smoke[n]−smoke[i])` once, ± apply, shared narrow → **also LSB-conservative** (a bonus the current `+=
  d_eff·dt·lap` truncated-Laplacian form does NOT give). `wind_sq` square in int64; `wind_diffusion_scale=50`
  and `d_smoke` constants pinned in Q16.16. **The n_smoke cliff** (`physics_engine.cpp:183-198`, the HARD
  cliff): `max_wind_sq` = the **NEW integer max reduction** over the Q16.16 wind field (order-free); square
  in int64; `d_eff_max`, `dt_stable = 1/(4·d_eff_max)` (reciprocal), `ceil_div`. A 1-ULP slip here flips the
  **substep count** → peers iterate differently → total desync (a naive within-substep digest misleads).
  **Owed: the `d_smoke` / `wind_diffusion_scale` retune** — the donor-cell numerical diffusion overlaps
  heavily with this wind-diffusion (research §4); retune both *down* to recover calm-air crispness (a retune
  was already owed from the `dt_scale²` removal, `05_smoke.md`). Q-S2-4.
- **S2b-4 — `sink_hop` → sink-velocity flux bias (the genuinely-new piece, research §5.1):** today
  `sink_hop` (`smoke_dynamics.cpp:225-277`) is a semi-Lagrangian gather that pulls toward the BFS breach
  direction (`sink_x/sink_y`) and **deliberately deletes mass** by sampling a 0 breach corner. **Reformulate
  as an extra advective velocity** `= sink_strength·sink_dir` added into `v_face` for the **same
  conservative flux gather**, run K× (`vent_hops=16`). The mass-deletion now happens *naturally* via the
  flux into a zeroed vacuum cell (no separate mass-deleting pass). This is a **behavioural rewrite, not a
  copy** — **in-scope (Q-S2-5)**, feel-gated A/B (does the room still vent at the right rate?).
- **S2b-5 — batch the 5 gas planes** through the identical kernel (they reuse everything; smoke is plane 1).

**Gate S2b:** P1 `tol=0.0`; **P2 — each of the 5 gas planes' `Σ` LSB-constant in a sealed room** (flux-form
makes smoke conservative for the *first time* — there is no conservation regression to protect, only a
property to gain; add a stress test like S1's flood test); feel A/B on the rendered smoke (crisp wisps,
right venting rate). The per-plane `.any()` (`physics_engine.cpp:208-242`) compares the **integer** field
`!= 0`, never a float bridge.

### S2c — atmosphere diffusion (RB-GS) + wind

The implicit diffusion (`diffuse_solve`, `atmosphere_solver.cpp:168-426`), run ×1 per tick.

- **mu** (`:183`): `mu = d_atm·dt` (`d_atm=200`, `mu≈8.33`) — scalar `mul_q16` per tick; `mu > 1e-8` →
  pinned Q16.16 compare.
- **RB-GS structure** (`:201-205`): the **confirmed 2-color schedule** is **order-independent by
  construction** (red `(x+y)&1==0` reads only black neighbours, no intra-color read-after-write) → identical
  on any architecture, two kernel launches + a barrier on GPU. The schedule is a *thing to pin*, **not** a
  desync. **The ONLY hazard is the per-cell divide** (`:257`).
- **The GS per-cell divide** (`:257`, **the S2c centerpiece**): `atmosphere[i] = (rhs[i]+μ·nb)/(1+μ·wsum)`.
  Replace with the **precomputed reciprocal `Dinv[i]`** (the NEW helper, §2.2 #3) in **residual/flux form,
  NOT the quotient form** (the quotient form has no fixed point under a truncating multiply at `μ·wsum≈33` →
  systematic leak; map §4): per cell,
  `atm[i] += mul( Σ_face mul(mul(mu, w_face), atm[n]−atm[i]) − (atm[i]−rhs[i]), Dinv[i] )`.
  Equal neighbours at the fixed point → the increment truncates to exactly 0 → drift-free. `Dinv` precomputed
  once per tick on changed cells (Q-S2-3: double-reciprocal-then-quantize vs integer Newton). **Note:** the
  `w∈{0,½,1}→shift` collapse (map M7) is **conditional on permeability being discrete** — Q4 kept
  permeability *continuous*, so the shift collapse may NOT apply; treat `mul(mu, w_face)` as a full Q16.16
  product unless a config audit confirms permeability is `{0,0.5,1.0}` at HEAD (flag in the impl).
- **GS-residual hook** (`:274-301`): a **read-only diagnostic** (`last_gs_residual`); nothing in-sim reads
  it. **Mirror it in integer** for the convergence-acceptance test (§4) — keep the float CPU oracle too.
- **vac_dist BFS** (`:322-355`): **already integer** (uint8 BFS) — no change.
- **eta + sponge BCs** (`:357-380`): `eta = min(breach_rate·dt, 1)` via `mul_q16` + Q16.16 min; the
  `atmosphere *= (1−eta)` / `(1−eta·0.5)` / `(1−eta·0.25)` are `mul_q16` + `shr_round0` (`>>1`/`>>2`). This
  is a **deliberate sink** (air vents to space) — NOT conserved here, by design (like temperature cooling);
  plain `mul_q16`/shift, no flux pair.
- **wind = −∇(atm+wave_p)** (`:413-420`): `total = atm + wave_p` (a 2-term sum; if wave_v/wave_p went Q24.8
  this is the cross-format read); `p_side = p_here + f·(total(n)−p_here)` via `mul_q16`; `wind_x =
  −(p_right−p_left)·0.5`, `wind_y = −(p_down−p_up)·0.5` via **`shr_round0`** (wind is signed). 2-term central
  diff. **The smoke bilinear renorm divide is GONE** (the flux-form switch in S2b deleted it — this is one
  of the §2.3 helpers we no longer need).

**Gate S2c:** P1 `tol=0.0`; **P2 — atmosphere bulk mass LSB-conserved** in a sealed region (the GS
redistribution + transfer hold; the vacuum/sponge BC is the *intended* sink exception); **the GS-residual
convergence check** (§4 — does the integer RB-GS converge within a stated factor of the float build's
residual). Feel: atmosphere settles right (no checkerboard, drains to vacuum at the right rate).

---

## 4. Gating (the acceptance contract)

Mirrors S1 §6, with two S2-specific additions (conservation across more fields; GS convergence):

- **P1 — within-config bit-identity** (`tol=0.0`, run twice, bit-identical via the field A/B harness incl.
  the S0 unit-state digest) **AND cross-config self-consistency** (vary `tps` / `tile_size_m` — the integer
  path stays internally consistent). Each sub-step self-matches at `tol=0.0`.
- **P2 — conservation, to the LSB:**
  - **atmosphere bulk mass** Σ constant in a sealed region (transfer + GS hold; vacuum/sponge is the
    intended sink exception).
  - **each of the 5 gas planes' Σ** constant in a sealed room — **flux-form makes smoke conservative for the
    first time** (today's SL tolerates ≤10% loss, `test_smoke_semilagrangian.py:181`); no regression to
    protect, a property to *gain*. **Add a smoke stress test like S1's flood test** (seal a room, settle
    many ticks, assert `Σ` bit-constant; a separate test asserts venting deletes mass *only* through breach
    faces).
  - the wave→atmosphere transfer **mass-neutral** (the rounded `mean_wp` — a biased mean is a DC leak).
- **GS-residual convergence check** (the separate "converges" claim, distinct from "deterministic"): mirror
  the `:274-301` residual hook in integer; assert the integer RB-GS's Linf residual is **within a stated
  factor of the float build's** on the stress scenarios (reuse the Patch-2 GS-residual hook). "Deterministic"
  and "converges" are tested separately — a drift-free-but-non-converging GS would pass P1 and fail this.
- **Feel-regression:** smoke still reads as crisp wisps (S2b SSIM A/B vs the float SL golden); atmosphere
  settles right; venting at the right rate (the `sink_hop` rewrite A/B). Erik's eye on each feel-gated step.
- **Goldens regenerated + version-bumped in the same commit** (the integer trajectory ≠ the old float
  exactly; smoke's golden changes the *most* — it's a scheme switch, not just a representation change).
- **Full suite green + both `--auto` exit 0.**

---

## 5. Open questions for Erik (the decisions)

- **Q-S2-1 — THE HEADLINE: smoke/gas advection.** **SWITCH** smoke + the 5 gas planes from semi-Lagrangian
  back-trace to **conservative flux-form (donor-cell + divide-free MC/minmod limiter, Family 3c)** —
  *recommended* — vs **keep-SL-made-conservative** (the Lentine per-source-weight=1 prototype, research §9
  alternative). The switch *deletes* the per-cell `sqrt`, three rounding modes, wall-clip march, and renorm
  divide; keeping SL drags all of them (incl. integer sqrt + integer bilinear) into scope. **Recommend:
  switch.**
  - **Q-S2-1b — the limiter:** **MC (two-slope, divide-free)** default (smooth-smoke look) vs **minmod**
    (safe fallback, slightly soft) vs **bare donor-cell** (zero-risk floor, ship in S2b-1, add limiter in
    S2b-2). NOT superbee (stair-steps — wrong look), NOT van Leer (a per-cell divide for ~5% over MC).
    **Recommend: MC, with bare-donor as the S2b-1 landing.**
- **Q-S2-2 — wave_v format exception.** **Measure peak |wave_v| in a blast first.** If post-dt `wave_v`
  stays inside ±32768 → keep **int32 Q16.16** with the **dt-before-narrow** int64 discipline at the kick
  (`:108`). If it overflows → **Q24.8 for wave_v alone** (wave_p stays Q16.16, the kick converts at the
  narrow). **Recommend: measure, default Q16.16, Q24.8 only if the measurement forces it.**
- **Q-S2-3 — the GS `Dinv` reciprocal method.** **(a) `/fp:strict` double reciprocal-then-quantize**
  (correctly-rounded, deterministic, simple, but keeps a float in the per-cell precompute → not GPU-pure)
  vs **(b) fixed-iteration integer reciprocal** (Newton/FixPointCS, pure integer, GPU-clean, more code).
  **Recommend: ship (a) on CPU now behind a `reciprocal_q16(denom)` signature so (b) drops in for the GPU
  port; record the precompute-float as a CPU-only artifact in the format tag.**
- **Q-S2-4 — the `d_smoke` / `wind_diffusion_scale` retune.** The donor-cell numerical diffusion overlaps
  heavily with the wind-coupled Laplacian (research §4) → both owe a *downward* retune to recover calm-air
  crispness (a retune was already owed from the `dt_scale²` removal). **This is owed**, scoped to S2b-3,
  feel-gated. Confirm the retune is in-scope for S2 (vs a follow-up tuning pass).
- **Q-S2-5 — confirm the `sink_hop` rewrite is in-scope.** The SL mass-deleting gather → a sink-velocity
  flux bias into the conservative face flux (research §5.1) is a *behavioural rewrite*, not a port.
  **Recommend: in-scope for S2b-4, feel-gated A/B.**
- **Q-S2-6 — freeze the int16(Q1.15) gas width now.** The 5 `[0,1]` gas planes are recorded as int16(Q1.15)
  in the format-version tag this session (ship int32 on CPU). Confirm the freeze so the CUDA buffers +
  digest schema are designed once. **Recommend: freeze.**
- **Q-S2-7 — `mean_wp` edge-flux retirement timing.** Ship the **rounded-mean stopgap** in S2a (correct +
  deterministic), retire to the local edge-flux transfer as a **follow-up commit** (it changes the
  transfer's spatial structure → feel-gated, not bit-compatible with the mean form). Confirm: stopgap now,
  retire later (vs do the edge-flux transfer directly in S2a). **Recommend: stopgap now.**

---

## 6. Risks

### 6.1 The flux-form switch is the biggest behaviour change of the whole arc
S1 was a *representation* change (the float and integer trajectories differ only at the LSB). **S2b is a
*scheme* change** — the smoke field will look measurably different from the float SL golden (the whole point:
it now conserves). The **feel-regression gate matters most here** of anywhere in the arc. Mitigations: ship
bare donor-cell first (S2b-1, conservation+determinism, no limiter) so the conservation milestone is
de-risked independently of the look; add the MC limiter (S2b-2) under an SSIM A/B; the research argues the
visual cost is partly pre-paid by the existing wind-diffusion and owned by the §6.1 render shader. If the MC
look disappoints, minmod or (last resort) van Leer's per-cell divide are the escalation path — but do **not**
fall back to the global mass-fixer (the `mean_wp`-class reduction the arc exists to kill).

### 6.2 The atomic 5-system landing is a single big step
The group must land together (wind feeds smoke within a tick, §1.1). To gate sub-steps **incrementally
even though they land together**, use an **internal float-bridge ordering** during development: keep S2c's
wind float while migrating S2b (S2b reads a dequantized wind), keep S2a's atmosphere float while migrating
S2c, etc. — so each sub-step's P1/P2 is tested in isolation against a float-bridged neighbour, then the
bridges are removed in the final merge and the *whole group* re-gated at `tol=0.0` cross-config. This mirrors
S1's W3/W5 bridge discipline, but *internal* to the group. The final merge's P1 must pass with **zero**
within-group float bridges (only the fire bridge remains).

### 6.3 The runtime cliffs (`n_smoke`) are a rank-2 desync vector
`n_smoke` depends on a runtime reduction (`max_wind_sq`); a 1-ULP slip flips the **substep count**, not a
field value → peers iterate a different number of times → total desync that a within-substep digest can
*miss*. The reduction is a `max` (order-free for integers) and the cliff arithmetic is `ceil_div`, but the
test must assert on the **substep count itself**, not just the post-loop field. (`n_wave` is the easy cliff —
config-constant.)

### 6.4 The GS `Dinv` precompute float (if Q-S2-3 picks (a))
The double-reciprocal-then-quantize keeps a float op in the per-cell precompute. It's deterministic on CPU
(correctly-rounded IEEE div + pinned quantize), but the **GPU port must match the CPU quantization
bit-for-bit** or the two builds diverge — the integer-Newton path (b) avoids this entirely. Designing the
helper signature so (b) can replace (a) without touching call sites is the mitigation.

### 6.5 The wave_v overflow (S2a)
`c_sq·lap` exceeds ±32768 before ·dt at `c_sq=4356`. If the int64-intermediate / dt-before-narrow discipline
is missed anywhere the kick touches, wave_v silently wraps → a blast desyncs. The Spike measurement
(Q-S2-2) and an explicit overflow assertion in the blast stress scenario are the guard.

### 6.6 The `mean_wp` rounding bias (S2a)
A signed truncated mean biases by `sign(sum)` → a DC drift into *every* atmosphere cell → a P2 defect that a
casual test (which checks magnitudes, not the integral) can miss. The round-to-nearest-even mean and a
dedicated "transfer is mass-neutral to the LSB" P2 test are the guard.

---

*Companion docs: `docs/s2_atmosphere_group_map.md` (the HEAD survey), `docs/s2_advection_research.md` (the
flux-form recommendation), `docs/fixed_point_migration_plan.md` §9 (locked resolutions),
`docs/s1_water_fixed_point_plan.md` + `cpp/src/fixed_point.h` (the shipped template). All line numbers
verified at HEAD 2026-06-24; re-verify before impl.*
