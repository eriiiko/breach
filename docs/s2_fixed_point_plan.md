# S2 — Atmosphere / Wave / Wind / Smoke / Gas → integer (Q16.16) — plan

**Status:** plan. The §5 headline (Q-S2-1, smoke advection) is now **RESOLVED — SAVE semi-Lagrangian,
integer-SL-everywhere** (see §5); the remaining §5 items are forward decisions awaiting Erik's OK.
The **second field migration** of the fixed-point arc and **the last big coupling group before fire (S3)**.
Built from `docs/s2_atmosphere_group_map.md` (the factual HEAD survey — every divide, transcendental,
reduction, conservation property and cliff), `docs/s2_advection_research.md` (the smoke-advection
recommendation — **note: its flux-form recommendation is now OVERRIDDEN, see §5/Q-S2-1**),
`docs/fixed_point_migration_plan.md` §9 (the locked resolutions), the shipped S1
toolkit `cpp/src/fixed_point.h` + `docs/s1_water_fixed_point_plan.md` (the conservative gather-once / ±-pair /
shared-narrow template), and the **committed integer-SL prototype** (`ceb601b`, branch `s2-advection-demo`,
`tools/s2_advection_demo/`) that empirically settled Q-S2-1.

**Unlike S1, this group is HARD:** it has a global reduction (`mean_wp`), two genuine dynamic divides that
remain (the GS per-cell divisor, the smoke bilinear renorm — both serviced by the *same* `reciprocal_q16`
helper), two runtime-derived integer cliffs (`n_wave`, `n_smoke`), and **five systems that couple inside one
tick** so they must migrate **atomically**. The plan's headline move is to **KEEP smoke/gas on
semi-Lagrangian and take it integer everywhere** — one Q16.16 SL field drives both visual and gameplay, so
there is no visual/gameplay drift. This was settled **empirically** (Q-S2-1, §5): the advection demo + the
integer-SL prototype (`ceb601b`) proved a Q16.16 SL is **bit-deterministic, a visual twin of the float SL,
and gently non-conservative** (the `>>16` truncation is a built-in mild decay), and Erik **accepted
deterministic non-conservation** — it is identical on every machine, so it is *behaviour, not desync*, with
**smoke decay as the tuning knob**. This **overrides** `s2_advection_research.md`'s flux-form recommendation.
The back-trace `sqrt` is avoided by a **sqrt-free DDA march** (no transcendental); the bilinear renorm divide
is folded into the shared `reciprocal_q16`.

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
| **gas** — 5 planes (white_smoke, **black_smoke**, poison, teargas, fuel_gas) | (5,h,w) f32 | no | **NO — integer-SL, deterministic-but-non-conservative** (accepted; decay is the knob) | `[0,1]`-clamped tracers; Q16.16 res ≈1.5e-5 ≪ perceptual; no overflow in the SL bilinear `mul_wide` int64 accumulate | **int16 (Q1.15) frozen, ship int32 on CPU** |
| **smoke** | **view of gas[BLACK_SMOKE]** | no | **NO (same — integer-SL non-conservative)** | *same storage as a gas plane* — black_smoke IS plane 1 | int16 (Q1.15) |

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
- `mul_q16` (`:81`) + `mul_wide` (`:89`) + `narrow` (`:96`) — every Q16.16 multiply, **the conservative
  flux gather** (gather wide, ± the same int64 to both cells, narrow once) for the wave Laplacian, the
  atmosphere GS stencil, and the smoke wind-diffusion Laplacian, **and the integer-SL bilinear sample**
  (the 4 corner-weight products accumulated in int64, then narrowed — `ceb601b`).
- `make_recip` / `recip_mul` (`:121-154`) — the *loop-invariant* divides (`dt_actual = sim_time/n`,
  `dt_stable`, `mu = d_atm·dt`). **NOT** valid for per-cell runtime divisors (the GS `Dinv`, the SL bilinear
  renorm) — `make_recip` is double-at-load for a single divisor; those use the per-cell `reciprocal_q16`
  (§2.2 #3).
- `shr_round0` (`:160`) — symmetric decay for signed fields (the sponge `*0.5`/`*0.25`, the wind `*0.5`).
- `scale_mag` (`:175`) — shrink-only signed scale: the wave absorb `×k` (k∈[0,1]). (No smoke outflow
  limiter — integer-SL is a back-trace gather, not a face-flux scheme, so there is no per-cell outflow sum to
  clamp.)
- `ceil_div` (`:189`) — the two S2 substep cliffs (n_wave, n_smoke). (The integer-SL DDA march is a
  dominant-axis cell-by-cell loop with an integer cap, not a `ceil_div` cliff.)
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

3. **Per-cell reciprocal `reciprocal_q16(denom)` — serves BOTH the GS `Dinv` AND the SL bilinear renorm.**
   The GS divisor `1 + μ·wsum` is **per-cell AND continuous** (Q4 kept permeability continuous, so `wsum` is
   not just `{0,½,1}` sums — a per-cell reciprocal is genuinely needed each tick); the integer-SL bilinear
   needs the *same* primitive for `1/wsum` at sealed corners (clamp `wsum` to a floor first). One helper, two
   call sites — the prototype (`ceb601b`) already uses the **3-step Newton reciprocal** (`r ← r·(2−wsum·r)`,
   seeded from a power-of-2 reciprocal) for the SL renorm; that is exactly path (b) below. `make_recip`
   cannot serve either (it's double-at-load for one divisor). **This is the locked design choice Q-S2-3:**
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

4. **Integer-SL advection machinery (the back-trace gather)** — three pieces, all proven in `ceb601b`:
   - **A sqrt-free DDA wall-clip march.** Back-trace the departure point in Q16.16, then step the **dominant
     axis cell-by-cell** (Chebyshev-distance march) toward the source, **stopping before a sealed tile** —
     so it can never tunnel a 1-cell wall, and it needs **no `sqrt`** for the march length (the transcendental
     the float SL had at `smoke_dynamics.cpp:64` is gone).
   - **An integer bilinear sample.** The 4 corner weights as Q16.16 products, accumulated in **int64**, then
     narrowed (`mul_wide`/`narrow`). Sealed corners contribute 0.
   - **The renorm `1/wsum` via `reciprocal_q16` (the shared §2.2 #3 helper)** — the prototype's 3-step Newton
     reciprocal `r ← r·(2−wsum·r)`, `wsum` clamped to a floor; the **same routine as the GS `Dinv`**, reused.
   Then a **pinned `>>16` truncation** (mirroring `mul_q16`), which is what makes the scheme *gently
   non-conservative* (the truncation bleed = a built-in mild decay). No flux pair, no limiter.

### 2.3 What integer-SL needs vs does NOT need

Integer-SL **DOES use the SL bilinear** (the 4-corner integer sample + the `reciprocal_q16` renorm). It does
**NOT** need an integer `sqrt` (the DDA march replaces the back-trace length `sqrt` at `smoke_dynamics.cpp:64`
with sqrt-free dominant-axis stepping), and it does **NOT** need the **flux-limiter machinery** (donor-cell
faces, the `min`/`max`/`minmod` limited correction flux, the outflow clamp) — those were the *flux-form*
route we did not take. (This inverts the earlier draft, which said we did not need the bilinear *because*
we were going flux-form; we kept SL, so the bilinear is in and the limiters are out.) The renorm divide is
not deleted — it is folded into the one shared `reciprocal_q16`.

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

### S2b — smoke + 5 gas (integer semi-Lagrangian — KEEP the scheme, take it integer)

**Q-S2-1 RESOLVED (§5): KEEP semi-Lagrangian, integer-SL-everywhere.** One Q16.16 SL field drives both
visual and gameplay → no visual/gameplay drift. The scheme is the *same* back-trace SL the float build runs
today (`smoke_dynamics.cpp`), re-expressed in Q16.16 exactly as the **proven prototype** (`ceb601b`, branch
`s2-advection-demo`, `tools/s2_advection_demo/`) — bit-deterministic, a visual twin of the float SL, and
gently non-conservative (the `>>16` truncation is a built-in mild decay). This **overrides**
`s2_advection_research.md`'s flux-form recommendation. There is **no donor-cell / no limiter / no outflow
clamp** anywhere in S2b. Authored as a gated sequence:

- **S2b-0 — representation:** quantize smoke + 5 gas planes to int32 Q16.16 (`[0,1]` tracers share the
  water/heat scale). The field **persists as int32 across ticks** (the SL reads last tick's int field, never
  a re-quantized float — see §6.1); float dequantize only for the renderer + the fire bridge.
- **S2b-1 — the integer-SL back-trace (the core, ported from `ceb601b`):**
  - **Back-trace** the departure point in Q16.16 from the quantized wind (`dt` is the loop-invariant
    `recip_mul` constant).
  - **The sqrt-free DDA wall-clip march** — step the **dominant axis cell-by-cell** (Chebyshev march)
    toward the source and **stop before a sealed tile**, so it cannot tunnel a 1-cell wall and needs **no
    `sqrt`** (this replaces the float back-trace length `sqrt` at `smoke_dynamics.cpp:64`).
  - **The integer bilinear sample** — 4 corner-weight products in Q16.16, accumulated in **int64**, then
    **narrowed**; sealed corners contribute 0.
  - **The renorm `1/wsum` via `reciprocal_q16`** (the shared §2.2 #3 helper — the prototype's 3-step Newton
    `r ← r·(2−wsum·r)`, seeded from a power-of-2 reciprocal, `wsum` clamped to a floor).
  - **A pinned `>>16` truncation** (mirroring `mul_q16`) — this is the gentle non-conservation (truncation
    bleed = mild decay).
  **Gate P1 (`tol=0.0`, run twice → bit-identical int32 field, the prototype's hard determinism assert) +
  the deterministic-non-conservation check (§4)** — total mass identical run-to-run, bounded/gently-decaying,
  never blowing up. *Not* a P2 conserve-to-LSB gate (SL does not conserve — accepted).
- **S2b-2 — `sink_hop` stays an SL-style integer breach-pull (NOT a flux bias):** `sink_hop`
  (`smoke_dynamics.cpp:225-277`) remains the semi-Lagrangian gather that pulls toward the BFS breach
  direction (`sink_x/sink_y`) and deletes mass by sampling a 0 breach corner — re-expressed in the *same*
  integer-SL machinery above (DDA march + integer bilinear + renorm), run K× (`vent_hops=16`). It is **not**
  reformulated as a sink-velocity flux bias (that was the flux-form route we did not take). Feel-gated A/B:
  does the room still vent at the right rate?
- **S2b-3 — wind-dependent diffusion (stays, integer) + the n_smoke cliff:** the Pass-A wind-coupled
  Laplacian (`smoke_dynamics.cpp:163-170`) stays as the diffusion step, taken **integer** (`mul`/`mul_wide`,
  `wind_sq` square in int64, `wind_diffusion_scale=50` and `d_smoke` pinned in Q16.16). **The n_smoke cliff**
  (`physics_engine.cpp:183-198`, the HARD cliff): `max_wind_sq` = the **NEW integer max reduction** over the
  Q16.16 wind field (order-free); square in int64; `d_eff_max`, `dt_stable = 1/(4·d_eff_max)` (reciprocal),
  `ceil_div`. A 1-ULP slip here flips the **substep count** → peers iterate differently → total desync (a
  naive within-substep digest misleads). **The advection-overlap retune is moot** (we kept SL, so there is no
  donor-cell numerical diffusion overlapping the wind-diffusion — the diffusivity is unchanged). A separate
  `dt_scale²`-removal retune from S1 may still be owed, tracked under Q-S2-4.
- **S2b-4 — batch the 5 gas planes** through the identical SL kernel (they reuse everything; smoke is
  plane 1). All 5 are integer-SL-advected — deterministic-but-non-conservative, accepted.

**Gate S2b:** P1 `tol=0.0` (run twice → bit-identical int32 field, per the prototype's checksum assert);
**NOT a P2 conserve-to-LSB gate for smoke/gas** — instead the **deterministic-non-conservation check** (total
mass per plane identical run-to-run, bounded and gently-decaying, never amplifying; the prototype showed ~50%
of peak kept in calm, ~85% in a blast) **+ a feel-regression vs the float SL** (reuse the prototype's
float-vs-int comparison — `scenario2_SLfloat_vs_SLint.png`: the filled, internally-structured look survives,
only faint edge-speckle in the ×6 diff). The per-plane `.any()` (`physics_engine.cpp:208-242`) compares the
**integer** field `!= 0`, never a float bridge. (Conservation-to-LSB is gated on **atmosphere** in S2a/S2c,
NOT on smoke — see §4.)

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
  diff. **The smoke bilinear renorm divide stays** (integer-SL kept it) — but it is the *same*
  `reciprocal_q16` helper as the GS `Dinv` (§2.2 #3), so it costs no new primitive.

**Gate S2c:** P1 `tol=0.0`; **P2 — atmosphere bulk mass LSB-conserved** in a sealed region (the GS
redistribution + transfer hold; the vacuum/sponge BC is the *intended* sink exception); **the GS-residual
convergence check** (§4 — does the integer RB-GS converge within a stated factor of the float build's
residual). Feel: atmosphere settles right (no checkerboard, drains to vacuum at the right rate).

---

## 4. Gating (the acceptance contract)

Mirrors S1 §6, with S2-specific additions (per-field conservation — atmosphere conserves to the LSB, smoke
does NOT; a deterministic-non-conservation check for smoke/gas; GS convergence):

- **P1 — within-config bit-identity** (`tol=0.0`, run twice, bit-identical via the field A/B harness incl.
  the S0 unit-state digest) **AND cross-config self-consistency** (vary `tps` / `tile_size_m` — the integer
  path stays internally consistent). Each sub-step self-matches at `tol=0.0`.
- **P2 — conservation, to the LSB — but PER FIELD (smoke is NOT in P2):**
  - **atmosphere bulk mass** Σ constant in a sealed region (transfer + GS hold; vacuum/sponge is the
    intended sink exception). **This is where the conserve-to-LSB gate lives.**
  - the wave→atmosphere transfer **mass-neutral** (the rounded `mean_wp` — a biased mean is a DC leak).
  - **NOT smoke / gas.** Integer-SL does **not** conserve (the `>>16` truncation is a deliberate gentle
    decay) — this is **accepted** (deterministic on every machine → behaviour, not desync; smoke decay is the
    tuning knob). So the 5 gas planes do **not** get a P2 conserve-to-LSB test. Instead they are gated on:
- **Deterministic-non-conservation check (smoke / gas, replacing P2 for these fields):** total mass per plane
  is **identical run-to-run** (the prototype's bit-identical-checksum assert), and **bounded / gently
  decaying / never blowing up** (the prototype showed ~50% of peak kept in calm, ~85% in a blast — never
  over-amplifying). Seal a room, settle many ticks, assert the per-tick mass trace is bit-identical between
  two runs and monotone-ish downward, not divergent.
- **Feel-regression vs the float SL (smoke / gas):** reuse the prototype's float-vs-int comparison
  (`scenario2_SLfloat_vs_SLint.png`) — the filled, internally-structured SL look survives the integer
  approximations (only faint edge-speckle in the ×6 diff). Erik's eye signs off; the `sink_hop` breach-pull
  still vents at the right rate (A/B).
- **GS-residual convergence check** (the separate "converges" claim, distinct from "deterministic"): mirror
  the `:274-301` residual hook in integer; assert the integer RB-GS's Linf residual is **within a stated
  factor of the float build's** on the stress scenarios (reuse the Patch-2 GS-residual hook). "Deterministic"
  and "converges" are tested separately — a drift-free-but-non-converging GS would pass P1 and fail this.
- **Goldens regenerated + version-bumped in the same commit** (the integer trajectory ≠ the old float
  exactly). Smoke stays on the *same SL scheme*, so its golden changes only at the representation level (the
  integer approximations + the `>>16` decay), not from a scheme switch — the look is preserved (the prototype
  proved it).
- **Full suite green + both `--auto` exit 0.**

---

## 5. Open questions for Erik (the decisions)

- **Q-S2-1 — THE HEADLINE: smoke/gas advection. RESOLVED.** **SAVE semi-Lagrangian —
  integer-SL-everywhere.** One Q16.16 SL field drives **both** visual and gameplay → no visual/gameplay
  drift; deterministic; **non-conservative-but-deterministic, with smoke decay as the knob.** This was
  decided **empirically** and **overrides** `s2_advection_research.md`'s flux-form recommendation: the
  advection demo + the integer-SL prototype (`ceb601b`, branch `s2-advection-demo`, `tools/s2_advection_demo/`)
  proved a Q16.16 SL is **(1) bit-deterministic** (run twice → bit-identical int32 field), **(2) a visual
  twin of the float SL** (the filled, internally-structured look survives the integer approximations — only
  faint edge-speckle in the ×6 diff, `scenario2_SLfloat_vs_SLint.png`), and **(3) gently non-conservative**
  (the `>>16` truncation acts as a mild built-in decay — keeps ~50% of peak in calm, ~85% in a blast, never
  over-amplifies). Erik **accepted deterministic non-conservation** — it is identical on every machine, so it
  is *behaviour, not desync*. The empirical proof + Erik's feel won over the flux recommendation.
  *(Q-S2-1b — the flux limiter choice — is removed; moot now that we keep SL.)*
- **Q-S2-2 — wave_v format exception.** **MEASURE FIRST.** Measure peak |wave_v| in a blast. If post-dt
  `wave_v` stays inside ±32768 → keep **int32 Q16.16** with the **dt-before-narrow** int64 discipline at the
  kick (`:108`). If it overflows → **Q24.8 for wave_v alone** (wave_p stays Q16.16, the kick converts at the
  narrow). **Recommend: measure, default Q16.16, Q24.8 only if the measurement forces it.**
- **Q-S2-3 — the `reciprocal_q16` method. LOCKED.** **CPU: `/fp:strict` double reciprocal-then-quantize**
  (`Dinv = quantize(1.0 / denom_real)`, correctly-rounded IEEE divide, deterministic for a given divisor,
  then quantized). **GPU: integer-Newton** (the prototype's 3-step `r ← r·(2−wsum·r)`), behind the same
  `reciprocal_q16(denom)` signature so it drops in without touching call sites. Record the precompute-float as
  a CPU-only artifact in the format tag. The `Dinv` is rebuilt **only on changed cells** each tick, keyed on
  `(mu | obstacles | is_wall | is_vacuum | permeability)`. (Same helper serves the SL bilinear renorm.)
- **Q-S2-4 — the `d_smoke` retune.** The **advection-overlap reason is GONE** — we kept SL, so there is no
  donor-cell numerical diffusion overlapping the wind-coupled Laplacian, and the diffusivity is unchanged. A
  separate **`dt_scale²`-removal retune from S1** may still be owed on its own merits (`05_smoke.md`); that is
  tracked here, but the advection-overlap retune is **moot**.
- **Q-S2-5 — `sink_hop` stays SL.** `sink_hop` remains the SL-style integer breach-pull (DDA march + integer
  bilinear + renorm, run K×), **not** reformulated as a sink-velocity flux bias. It is a port into the
  integer-SL machinery, feel-gated A/B (right venting rate). In-scope for S2b-2.
- **Q-S2-6 — freeze the int16(Q1.15) gas width now. FREEZE.** The 5 `[0,1]` gas planes are recorded as
  int16(Q1.15) in the format-version tag this session (ship int32 on CPU) so the CUDA buffers + digest schema
  are designed once.
- **Q-S2-7 — `mean_wp` edge-flux retirement timing. LOCKED.** Ship the **integer-mean stopgap** in S2a
  (correct + deterministic, the easier path), retire to the local edge-flux transfer as a **follow-up commit**
  (it changes the transfer's spatial structure → feel-gated, not bit-compatible with the mean form).

---

## 6. Risks

### 6.1 The int32 smoke field must PERSIST across ticks (the real S2b risk)
We KEEP semi-Lagrangian, so the smoke **look is preserved** — there is no scheme switch and no feel
regression on the look (the prototype proved the integer SL is a visual twin of the float SL). That risk is
**gone**. The real residual risk is plumbing: for end-to-end determinism the **int32 smoke/gas field must
persist across ticks** and be advected/diffused entirely in integer (the SL must read last tick's int field,
not a re-quantized float) — or, if any float boundary remains during migration, it must be a **pinned
float↔int boundary**. The S2 migration does this anyway (it's the whole point), and the advection step itself
is **proven deterministic** (`ceb601b`, run-twice bit-identical assert). The non-conservation is **accepted**:
deterministic on every machine, with the `>>16` decay as the tuning knob — behaviour, not desync.

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

### 6.4 The `reciprocal_q16` precompute float on CPU (Q-S2-3 is locked to CPU=(a), GPU=(b))
The locked CPU path (double-reciprocal-then-quantize) keeps a float op in the per-cell precompute. It's
deterministic on CPU (correctly-rounded IEEE div + pinned quantize), but the **GPU port (integer-Newton, the
prototype's 3-step routine) must match the CPU quantization bit-for-bit** on the same divisors or the two
builds diverge. The shared `reciprocal_q16(denom)` signature (one call site for both the GS `Dinv` and the SL
bilinear renorm) is what lets (b) replace (a) without touching call sites — that's the mitigation. A
cross-path equality test (CPU (a) vs GPU (b) on a sweep of divisors, to a stated reciprocal precision) is
owed before the GPU port.

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
flux-form recommendation — **OVERRIDDEN by Q-S2-1; we keep integer-SL, decided empirically by the `ceb601b`
prototype**), `docs/fixed_point_migration_plan.md` §9 (locked resolutions), `docs/s1_water_fixed_point_plan.md`
+ `cpp/src/fixed_point.h` (the shipped template), and the integer-SL prototype `tools/s2_advection_demo/`
(`ceb601b`, branch `s2-advection-demo`). All line numbers verified at HEAD 2026-06-24; re-verify before impl.*
