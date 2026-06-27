# S3 — Fire → integer (Q16.16 / int16) — plan

**Status:** plan, 2026-06-26. The **THIRD and FINAL field migration** of the fixed-point arc.
S1 (water) and S2 (atmosphere/wave/wind/smoke/gas group) are both **shipped to main** (HEAD `9e0e456`).
After S3 there is **NO float bridge left inside the sim** except the documented render/cosmetic ones.

Built from `docs/fixed_point_migration_plan.md` (the master plan + the §9 LOCKED resolutions),
`docs/s2_fixed_point_plan.md` + `docs/s2_atmosphere_group_map.md` (the structure + rigor this mirrors),
the **shipped** `cpp/src/fixed_point.h` toolkit (richer than the master plan's §2 implies — it already
carries `reciprocal_q16`, `mean_sum`/`mean_round`, `scale_mag`, `ceil_div`, `tan_poly`), and the three
production templates: `cpp/src/temperature_solver.cpp` (int32 Q16.16, the R2 difference-shift +
sign-symmetric cooling idioms; **fire reads `temperature` from here**), `cpp/src/smoke_dynamics.cpp`
(int32 Q16.16 SLint from S2b; **fire writes smoke here**), and the S2c integer atmosphere (**fire reads +
writes `atmosphere` here**).

All HEAD line numbers verified 2026-06-26 against `cpp/src/fire_simulation.{cpp,h}`,
`cpp/src/physics_engine.cpp` (`step_tail` :29-136, the fire bridge :89-133), `src/simulation/combat.py`
(`apply_temperature_ignition` :253-352). Re-verify before impl.

> **One-line orientation.** Fire is **one C++ TU and one Python function**, both small, both currently
> the LAST float consumers. The C++ `FireSimulation::step` (`fire_simulation.cpp:25-159`) is a single
> per-cell logistic update with **one genuinely new determinism problem the arc has never solved before:
> a per-cell transcendental** (`W = sqrt(wind_x²+wind_y²)`, `:86`). Everything else in fire is a mechanical
> port of idioms already shipped in `fixed_point.h`. The Python `apply_temperature_ignition`
> (`combat.py:253`) is a SECOND, separate fire-write path that **still dequantizes atmosphere to float**
> for its O2 mean (`:330-344`) — a Python float bridge that decides *who ignites*, in scope for S3.

---

## 1. Scope — what converts, what doesn't, and why this collapses the last bridge

### 1.1 The fire bridge today (what S3 deletes)

S2 left exactly ONE float bridge open inside the sim — the fire bridge — and it lives in
`physics_engine.cpp::step_tail` (`:89-133`), NOT in the fire TU itself:

```
// step_tail, per tick, today (the bridge S3 collapses):
for i: atm_f_[i]    = dequantize_f(atmosphere[i]);   // int32 -> float   (FIRE BRIDGE)
       wind_x_f_[i] = dequantize_f(wind_x[i]);
       wind_y_f_[i] = dequantize_f(wind_y[i]);
fire.step(fire_field, atm_f_, smoke_field/*int32*/, wall_hp, temperature/*int32*/,
          wind_x_f_, wind_y_f_, ...);                // float fire reads float atm/wind, writes int smoke
for i: atmosphere[i] = quantize(atm_f_[i]);          // float -> int32   (re-quantize the plume)
temperature.step(..., atm_f_/*float, read-only*/);   // temp reads the SAME float scratch
```

So fire **already half-lives in the integer world**: it reads `temperature` as int32 (`:63`), writes
`smoke` as int32 (`:126-131`, quantize-then-add, the S2b discipline), and the orchestrator round-trips
`atmosphere`/`wind` through float scratch only because the fire TU itself is still float. **S3 makes
`FireSimulation::step` integer end-to-end**, which deletes the three `atm_f_`/`wind_*_f_` dequantize
loops, the re-quantize loop, and the `mutable std::vector<float>` bridge buffers (`physics_engine.h:54-56`)
— and lets the temperature pass read the int32 `atmosphere` directly (its `atmosphere` arg becomes int32,
the cooling vacuum-exposure threshold becomes a Q16.16 compare).

### 1.2 CONVERT to integer (the synced fire state)

| Field | Today | Range | Signed | Conserved | Q-format | GPU width (frozen now, Q5) |
|---|---|---|---|---|---|---|
| **fire** (intensity I) | `float (h,w)`, `gamemap.py` | **[0,1] clamped** (`:99,153`) | no | no (logistic) | **int32 Q16.16 on CPU** (`[0,1]` tracer, δ≈1.5e-5 ≪ perceptual) | **int16 (Q1.15)** — frozen in the format tag, ship int32 on CPU |
| **wall_hp** (fuel F source) | `float (h,w)` | 0..~`fuel_ref`=60; HP is small integers (wood 30) | no | no (depleted) | **int32 Q16.16** (physical, >1 range) — see Open Q3 | int32 |
| **atmosphere** plume write | int32 Q16.16 (S2c) | already integer | no | yes (bulk) | already int32 — fire's plume += becomes an **integer deposit** | int32 |
| **smoke** emission write | int32 Q16.16 (S2b) | already integer | no | no (SLint) | already int32 — emission already quantize-then-add (`:126`) | int16 (Q1.15) |
| **temperature** read | int32 Q16.16 (shipped) | already integer | yes (ΔT) | no | already int32 — fire dequantizes-on-read today; reads int directly after | int32 |
| **wind_x/wind_y** read | int32 Q16.16 (S2c) | gradient O(0.1–1), shockwave spikes | yes | no | already int32 — fire dequantizes-on-read today; reads int directly after | int32 |

**The only NEW storage migration is `fire` itself** (and a decision on `wall_hp`, Open Q3). Atmosphere,
smoke, temperature, wind are **already int32** — fire just stops dequantizing them.

### 1.3 STAYS FLOAT — the render/cosmetic boundary (out of scope, the documented exceptions)

- **fire GLOW / render colour / flicker** — the synced/local rule (master plan §1.1): fire **intensity**
  (synced gameplay state) → integer; fire's *cosmetic* glow/colour is local, stays float, dequantized at
  the render boundary (one source of truth = the int field). Audit `physics_runner.py::cast_fire_heat`
  during impl: the heat-ray `LightSource` params `range = 2 + 2·I`, `intensity = 0.3 + 0.7·I` (engine/06
  §"Fire as a light source") read fire intensity — confirm they only touch `light_rgb`/ray params (local,
  float-OK) and the **`heat` deposit, which is already Q16.16 saturating-add** (`raycaster.cpp`, shipped).
  The fire→heat ray path is already integer on its deposit side; the ray *march* itself is render/heat
  infrastructure outside S3 (it is the same `sin/cos` ray path flagged for a later pass in the master
  plan §5.2 — NOT pulled into S3; fire only *enumerates burning tiles* and casts, it does no new ray math).
- **render-only fields** — `light_rgb`, `smoke_glow`, etc. unchanged.

### 1.4 The Python fence (Q2 — the boundary, stated precisely)

Two Python functions touch fire-adjacent synced state. Per the **Q2 LOCKED** decision (master plan §9),
combat-side HP/damage math **stays Python integers for now** (Python scalar int/float `+−×÷√` is
cross-machine reproducible — no FMA, no transcendental jitter; → C++ when ML training starts). S3's job is
to (a) make the one Python path that *writes fire* deterministic to match the integer sim, and (b) NOT
re-litigate the HP fence. Precisely:

- **`apply_temperature_ignition` (`combat.py:253-352`) — IN SCOPE.** This is a SECOND fire-write path
  (`fire = max(fire, ignition_seed)`, `:351`). Its temperature compare is **already integer Q16.16**
  (`:317`, the good pattern). BUT its O2 gate **dequantizes atmosphere to float32** and takes a float mean
  (`atm = dequantize_f32(gmap.atmosphere)` `:330`, `sum_atm/safe_count >= o2_threshold` `:331-344`). That
  float mean **decides who ignites** → it is synced gameplay state on a float bridge. **Fix in S3:** do the
  O2 mean in integer on the int32 `atmosphere` (the exact same int reduction the C++ fire uses for its `P`
  — §2.2). Once `fire` is int32 (§1.2), `ignition_seed` quantizes once and the `np.maximum` write is an
  integer max. (Note the fire→int means `gmap.fire` dtype flips — every Python reader of `gmap.fire`,
  incl. `cast_fire_heat` and the renderer, gets a dequantize boundary; audit in S3a.)
- **`apply_environmental_damage` / `apply_blast_damage` (`combat.py:110-247`) — OUT of S3 (Q2 fenced).**
  These read the int32 `heat` field and do float HP math → `current_hp`, kill events. They are
  lockstep-critical synced state but **Q2 keeps them Python-float for now**; the owed work is the
  **harness digest extension to the unit-state surface** (HP + hit/kill events), tracked as an S0-class
  deliverable in the master plan §6.1.3 — NOT fire-field work, but flagged here because S3 is where the
  fire→heat→damage loop closes and the leak becomes exercisable. (Open Q5.)

After S3 lands: the entire sim field path (water + atmosphere/wave/wind/smoke/gas + fire) is integer; the
only remaining float in the synced path is the Q2-fenced Python combat HP math (documented, harness-watched).

---

## 2. The Q16.16 idioms reused + the ONE new helper

### 2.1 Reused verbatim from `fixed_point.h` (shipped)

- `quantize` / `dequantize_f` (`:59-74`) — the render boundary + the `ignition_seed`/param load casts +
  the (now-deleted) fire bridge's last use.
- `mul_q16` (`:81`) + `mul_wide` (`:89`) + `narrow` (`:96`) — **the whole logistic update.** The 6-factor
  `grow` product and the `die` terms become chained `mul_wide` in int64, narrowed once at the
  `I += dt·(grow−die)` write (M2 — the Q16.48 chain, §3 S3b).
- `reciprocal_q16` (`:208`, the per-cell Newton-4-iter, **already shipped + validated for both the GS
  Dinv and the SL renorm**) — fire's per-cell divides reduce to this **only if** any survive after the
  divide-by-config audit (§2.3); most fire divides are by **config constants** → load-time reciprocals, not
  this. Listed because it is the fallback for `P = sum_atm/count` if that count is treated per-cell (it is
  small int 0..4 → a tiny LUT is better, §2.2).
- `mean_sum` / `mean_round` (`:312-327`) — **directly applicable to fire's `P` neighbour-mean** (`:72-82`)
  and to `apply_temperature_ignition`'s O2 mean: an int64 sum over the open-neighbour mask + a rounded
  integer mean. The count here is tiny (0..4), so see §2.2 for the cheaper specialization.
- `shr_round0` (`:160`) / `scale_mag` (`:175`) — signed-magnitude shifts; fire's fields are mostly `[0,1]`
  (unsigned), so these are needed only where a signed intermediate appears (the `grow − die` difference is
  signed — but it is a plain subtraction of two narrowed Q16.16 values, no shift).
- (`ceil_div` `:189` — see §4: fire has **no substep loop** today, so no cliff — but flag it if S3 adds one.)

### 2.2 The fire `P` neighbour-mean — `mean_sum`/`mean_round` or a tiny reciprocal LUT

`P = (count>0) ? sum_atm/count : 0` over the open (non-wall, non-vacuum) 4-neighbours (`:72-82`). `count ∈
{0,1,2,3,4}`. Two clean integer options, both deterministic:
- **`mean_sum`/`mean_round`** (the shipped helper) — int64 sum of the (now int32 Q16.16) neighbour
  atmosphere over the bool mask, then `mean_round(sum, count)` (round-to-nearest, sign-symmetric, no
  pre-shift — atmosphere is already Q16.16). **Recommended** — it is the exact primitive, already validated,
  and the same one `apply_temperature_ignition` should use, so the C++ fire and the Python ignition share
  ONE mean definition (they MUST agree — the design note `combat.py:276-281` requires the ignition O2 check
  and the fire O2 check be the same predicate, else a tile ignites into a state the fire immediately kills).
- A **5-entry reciprocal LUT keyed on count** (`recip[count]` in Q16.16, `1/1,1/2,1/3,1/4`) + `mul_q16`.
  Marginally cheaper (no per-cell divide), but `mean_round` is already there and carries the rounding
  contract. **Use `mean_sum`/`mean_round`** unless profiling later says otherwise.

### 2.3 Re-audit EVERY divide-by-config in fire (master plan M7/m7 — the divides §5 must not miss)

Fire's divides are almost all **by config constants** → precompute a reciprocal/shift at load (the
temperature `heat_inv_shift`/`face_shift` pattern; `make_recip`/`recip_mul` from `fixed_point.h:121-154`),
NOT a runtime `reciprocal_q16`:

| Site | Expression | Divisor | Idiom |
|---|---|---|---|
| `:48` | `inv_temp = 1.0 / temp_scale` | const (65536) | **load-time** scalar; `T = mul_q16(temperature[i], inv_temp_q)` — but note `temp_scale==FP_ONE`, so `T` in game-units IS `temperature[i]` already in Q16.16 (the divide is the identity at the shared scale — confirm + simplify). |
| `:67` | `F = clamp01(wall_hp / fuel_ref)` | const (60) | **load-time** `recip_fuel_ref`; `F = clamp01(recip_mul(wall_hp, r))`. |
| `:89` | `hot = clamp01((T − fire_T_ext)/fire_T_span)` | const (150) | **load-time** `recip_T_span`; the `(T − fire_T_ext)` is a signed Q16.16 subtract, then `recip_mul`, then `clamp01`. |
| `:90` (in `smoothstep`) | `t = (x − edge0)/(edge1 − edge0)` | const (`P_full−P_min`=0.4) | **load-time** `recip_P_span`; the smoothstep edges are config constants (§3 S3b). |
| `:82` | `P = sum_atm / count` | **runtime** (count 0..4) | `mean_round` (§2.2) — the ONE non-constant divide, and it is a tiny-int divide, not a per-cell continuous one. |
| `:113` | `atmosphere[i]/p_expand_ref` | const (1.30) | **load-time** `recip_p_expand_ref`. |

**No `reciprocal_q16` per-cell continuous divide exists in fire** (unlike S2c's GS). This is a simpler
divide profile than S2.

### 2.4 The ONE genuinely new primitive: a deterministic integer `sqrt` (the FIRST per-cell transcendental of the arc)

`W = sqrt(wind_x² + wind_y²)` (`fire_simulation.cpp:86`) is the **first per-cell transcendental the
fixed-point arc has had to solve.** S1's water `sqrt` is a **load-time constant** (`water_solver.cpp:30`,
computed once in double — IEEE sqrt is correctly-rounded → bit-identical, the "load-time sqrt is free"
lesson). S2's smoke **avoided** its back-trace `sqrt` entirely via the sqrt-free DDA march
(`smoke_dynamics.cpp`). So nothing in `fixed_point.h` computes a per-cell sqrt yet — **S3 must add it.**

- **What it is:** `W = sqrt(wx² + wy²)` per burning cell. `wx,wy` are Q16.16 signed; `wx²+wy²` is an int64
  Q.32 sum-of-`mul_wide` (the `max_wind_sq` reduction in `physics_engine.cpp:230-235` already builds exactly
  this int64 Q.32 quantity — reuse the pattern). `W` is then Q16.16.
- **The helper:** `sqrt_q16(int64 x_q32) -> q16` — a **fixed-iteration integer sqrt** (the FixPointCS /
  digit-recurrence algorithm the master plan §5.2 names). **Fixed iteration count, not `while(converged)`**
  → branch-identical across all lanes/architectures (master plan IDEA 12). The classic shape: integer
  bit-by-bit (`isqrt`) on the int64 radicand gives `sqrt(wx²+wy²)·2^16` directly when the radicand is the
  Q.32 sum (since `sqrt(2^32)=2^16`) — i.e. a **plain `isqrt64` of the Q.32 value yields the Q16.16 result**,
  no rescale. `isqrt64` (binary digit recurrence, 32 fixed iterations, pure integer shifts/compares) is
  fully deterministic and has **no rounding-mode ambiguity** (it returns floor(√), exact). This is the
  cleanest possible transcendental: **floor of an exact integer sqrt, no LUT, no poly, no libm.**
- **Why floor-isqrt and not a LUT/poly:** isqrt is *exact* (correctly-rounded-down by construction), needs
  no committed data file, and is identical on CPU and any future `__device__` port (the master plan §5.3
  contract: same integer algorithm both sides). A LUT would be a committed-data-file liability for zero gain.
  Add `sqrt_q16` to `fixed_point.h` next to `tan_poly`, with the same unit-test discipline
  (`for sampled int64 in-range: sqrt_q16(x)² ≤ x < (sqrt_q16(x)+1)²`).
- **Open Q1:** floor-isqrt truncates `W` toward 0 by up to ~1.5e-5 (1 LSB). `W` feeds `(1+k_wind_fan·W)`
  and `k_wind_strip·W·…` — both gentle, non-conserved, gain-bounded terms. The truncation is a deterministic
  ~1-LSB bias on a tuning term; **recommend floor-isqrt (exact, simplest), accept the bias** (it is
  perceptually invisible and deterministic-→-behaviour-not-desync). Round-to-nearest isqrt is available if a
  feel-check ever wants it (one extra compare), but it is unnecessary. Flagged for Erik because it is a
  genuine (tiny) behaviour choice, not a mechanical conversion.

### 2.5 `smoothstep` and `clamp01` — pinned fixed-point definitions (M2)

`clamp01(v)` → integer clamp to `[0, FP_ONE]` (trivial, exact). `smoothstep(edge0,edge1,x)`
(`fire_simulation.cpp:18-23`) → `t = clamp01(recip_mul(x−edge0, recip_span))` then the Hermite cubic
`t·t·(3−2t)` as three `mul_q16` + a constant — `3.0` is `3<<16`, `2t` is `t<<1`. **The multiply tree order
must be pinned** (master plan §2.4 fire row): compute `t2 = mul_q16(t,t)`, `three_minus = (3<<16) − (t<<1)`,
return `mul_q16(t2, three_minus)`. Define once, normatively, in the fire TU (or `fixed_point.h` if reused).
Edges are config constants → the span reciprocal is load-time (§2.3).

---

## 3. Sub-steps (each its own gated commit on an `s3-fire-fixedpoint` branch)

The fire TU is small and the couplings are few, so S3 is **two real sub-steps + a closer**, not five. The
authored order migrates the **producers fire reads (already integer) → fire's own state → the consumers fire
writes (already integer) → the Python ignition twin → the bridge deletion.**

### S3a — `fire` field representation + the Python ignition twin

Flip `fire` storage to int32 Q16.16 and make BOTH fire-write paths integer-consistent, WITHOUT yet
converting the C++ logistic math (keep the C++ `step` reading a dequantized fire for one commit, behind a
temporary internal float bridge — the S2 internal-bridge discipline, master plan §6.2 / s2 §6.2).

- **`gmap.fire` → int32 Q16.16** (`gamemap.py`). `g.fire[8,8] = 0.8` seeds become `quantize_scalar(0.8)`
  (the harness already does this for smoke, `field_ab_harness.py:106` — mirror it for fire `:107-108`).
- **`apply_temperature_ignition` (`combat.py:253`) → integer O2 mean + integer fire write:**
  - O2 gate: replace the `dequantize_f32(atmosphere)` + float mean (`:330-344`) with an **integer
    neighbour-sum + `mean_round`-equivalent** on the int32 atmosphere (NumPy int64 accumulate over the same
    N/S/E/W shifted slices, `safe_count` guard kept). The threshold `o2_threshold=0.60` quantizes once.
    This makes the ignition O2 predicate **bit-identical to the C++ fire's `P` gate** (§2.2) — the design
    note's required invariant (`:276-281`).
  - fire write: `ignition_seed=0.1` → `quantize_scalar(0.1)`; `np.maximum(gmap.fire, where(ignite, seed_q,
    gmap.fire))` is an **integer max** (exact, order-free).
- Every Python reader of `gmap.fire` gets a dequantize at its boundary (renderer, `cast_fire_heat`'s `range
  = 2 + 2·I` / `intensity = 0.3 + 0.7·I` — those are LOCAL/cosmetic + the heat payload, float-OK; audit).

**Gate S3a:** P1 self-match `tol=0.0` (fire field bit-identical run-to-run); the ignition path exercised
(a config where temperature crosses `ignition_temp` with O2 present — seed a hot flammable tile); the
ignition O2 gate proven equal to the C++ fire O2 gate on a shared cell (a targeted unit test). Golden
regenerated (fire dtype flip changes the digest representation). **NOT** a behaviour change yet (the C++
math still runs in float behind the internal bridge), so the feel is unchanged — this commit is pure
representation + the Python twin.

### S3b — the C++ logistic update in integer (the core, M2)

Convert `FireSimulation::step` (`fire_simulation.cpp:55-159`) to read/write integer end-to-end. Atmosphere
and wind args become `const int32_t*` (fire reads them directly — the bridge dequantize in `step_tail`
goes away in S3c once this lands). Per burning flammable cell:

- **`I = fire[i]`** — int32 Q16.16; the `I <= 0` skip is an integer compare; the `max_fire < 0.001f`
  early-exit (`:44-46`) becomes an **integer max + a pinned Q16.16 threshold compare** (`max_fire_q <
  quantize(0.001)`), an order-free integer max reduction.
- **`T` (the `hot` gate input):** float fire computes `T = temperature[i]/temp_scale` (real game-units,
  ~300) then compares against `fire_T_ext=350`. In integer we keep `T` as Q16.16: since `temp_scale ==
  FP_ONE == 65536`, the raw `temperature[i]` already equals `quantize(realΔT)`, so the dequantize-then-
  requantize-for-compare cancels — **just compare `temperature[i]` directly against `fire_T_ext_q =
  quantize(350)`** (and use raw `temperature[i]` as the Q16.16 `T` in the `hot` ramp). Confirm
  `temp_scale==FP_ONE` holds (it is the shipped default) and drop the divide. `temperature` is signed
  (ΔT) — `T` may be negative; `hot`'s `clamp01` handles it.
- **`F = clamp01(recip_mul(wall_hp[i], recip_fuel_ref))`** — load-time reciprocal (§2.3); `wall_hp` is
  Q16.16 (Open Q3).
- **`P = mean_round(sum_atm, count)`** over open neighbours (§2.2) — the int64 sum + rounded mean.
- **`W = sqrt_q16(mul_wide(wind_x[i],wind_x[i]) + mul_wide(wind_y[i],wind_y[i]))`** — the new isqrt helper
  (§2.4) on the int64 Q.32 radicand.
- **`hot = clamp01(recip_mul(T − fire_T_ext_q, recip_T_span))`**; **`o2 = smoothstep_q(P_min_q, P_full_q,
  P)`** (§2.5); **`avail = mul_q16(F, o2)`**.
- **`grow`/`die` — the 6-factor logistic in int64 (M2):** chain the products as `mul_wide`/`mul_q16`
  keeping a **wide int64 (Q16.48-style) intermediate, narrow ONCE** at the final
  `I_next = clamp01(I + mul_q16(dt_q, grow − die))`. **Pin the multiply tree order explicitly** — the master
  plan §2.4 flags this exact site: the chained truncation of `k_grow·avail·hot·I·(1−I)·(1+k_wind_fan·W)`
  depends on association order; fix it (e.g. left-fold, narrowing only at the end). `(1−I)`, `(1−avail·hot)`,
  `(1−I)·I` are Q16.16 subtractions/products. `dt` is the tick length → a Q16.16 scalar.
- **`I_next < I_min → 0`** snap-extinguish — integer compare (`I_min_q = quantize(0.02)`).
- **The discrete burning↔extinguishing flip is a control-flow OUTPUT that must be bit-deterministic**
  (master plan §2.4 fire row): the integer logistic must cross `I_min` on the identical tick on every peer.
  The whole chain being integer guarantees it; the gate (below) asserts it via the field digest.
- **Plume deposit (`:109-115`):** `atmosphere[i] += mul_q16(fire_pressure_gain_q, mul_q16(I, (FP_ONE −
  recip_mul(atmosphere[i], recip_p_expand_ref)))) · dt_q`, guarded `gain > 0`. This is an **integer deposit
  into the conserved atmosphere** — but it is a fire SOURCE (non-conserved by design, like the sponge sink),
  so a plain `mul_q16` round-trip is bit-safe (no flux pair needed; the master plan §1.1 and the existing
  re-quantize comment `physics_engine.cpp:111-115` already establish this). **Round-to-nearest the deposit**
  (the S2a/S2c unbiased-deposit lesson) so a long firestorm does not accumulate a truncation bias into
  atmosphere — use a round-half add on the final narrow, NOT plain `mul_q16` toward −∞, for this *deposit*
  (contrast the conservative flux pairs, which want the cancelling truncation).
- **Smoke emission (`:122-135`):** **already integer** (quantize-then-add, `:126`). The only change: the
  `smoke_emission·dt·I` rate currently quantizes a float `I`; now `I` is int32, so compute
  `delta_q = mul_q16(mul_q16(smoke_emission_q, dt_q), I)` (or quantize the product) — round-to-nearest the
  emission deposit (unbiased, same lesson). Order-free integer add into neighbours, unchanged.
- **Wall burn-through (`:137-148`):** `wall_hp[i] -= mul_q16(mul_q16(wall_damage_q, dt_q), fire[i])`;
  `wall_hp <= 0` is an integer compare; the destroyed-tile list is the control-flow output (deterministic
  by the integer compare). `fire[i] = 0` on burn-through, exact.
- **Final clamp (`:150-156`):** `fire` clamps to `[0, FP_ONE]` (integer), smoke already does.

**Gate S3b:** P1 self-match `tol=0.0`; the **burning↔extinguishing flip tick is bit-identical** (assert the
fire digest across two runs through an ignite→firestorm→starve→extinguish trajectory — the discrete output
is the determinism-critical thing); **the plume deposit into atmosphere is unbiased** (mean signed error ≈ 0
over a long firestorm — the deposit-rounding test, distinct from a conservation test since the plume is a
source); no int64→int32 overflow in the logistic chain under a shockwave-fanned firestorm stress scenario
(debug assert at every narrow). **Behaviour changes vs the float golden** (integer truncation ≠ IEEE) →
regen golden + **feel-gated A/B**: fire ignites, fans under wind, blows out, and extinguishes on
visually-identical ticks (the firestorm scenario rendered to PNG, SSIM/per-pixel diff, committed artifact).

### S3c — collapse the fire bridge (the closer)

Delete the float bridge now that fire is integer:

- **`physics_engine.cpp::step_tail`:** remove the `atm_f_`/`wind_x_f_`/`wind_y_f_` dequantize loops
  (`:98-105`), pass `atmosphere`/`wind_x`/`wind_y` (int32) straight to `fire.step`; remove the re-quantize
  loop (`:115`). The plume write is now an in-place integer `atmosphere[i] +=` inside the fire TU.
- **The temperature pass (`:130-133`):** its `atmosphere` arg (currently the float scratch, read-only for
  the vacuum-exposure threshold `atmosphere[ni] < o2_vacuum_thresh`, `temperature_solver.cpp:132`) becomes
  the **int32 atmosphere**; the threshold `o2_vacuum_thresh=0.3` becomes a Q16.16 compare constant
  (`quantize(0.3)`). This is the LAST consumer of the float bridge — converting it deletes the
  `temperature_solver.cpp` float-atmosphere read too (the `const float* atmosphere` arg → `const int32_t*`).
  **Note:** this touches `temperature_solver.{h,cpp}` signatures — a small, mechanical change, but it means
  the temperature TU's last float input is gone (it becomes fully integer, matching its already-integer
  temperature/heat fields).
- **Remove the bridge scratch buffers** `atm_f_`/`wind_x_f_`/`wind_y_f_` from `physics_engine.h:54-56`
  (the `wave_p_f_` buffer **stays** — the water head/ripple bridges still use it, master plan / S1, those
  are separate documented bridges retired by a later water/atmosphere unification, NOT S3).
- **The per-TU CI ratchet (master plan §6.3 S4):** add `fire_simulation.cpp` and the now-integer
  `temperature_solver.cpp` to the "no float/double/`/fp:fast` in the deterministic path" CI check, so a
  later patch cannot reintroduce a float into migrated fire/temperature. (Legit exception: the render/glow
  boundary, the load-time bakes.)

**Gate S3c:** P1 self-match `tol=0.0` with **ZERO float bridges inside the sim** (the whole field path —
water + S2 group + fire — is integer; only the Q2-fenced Python combat HP math remains, documented); the
P2 conservation tests (atmosphere bulk mass) still green (the plume deposit is the intended source
exception); full suite green + both `--auto` exit 0; the cross-config matrix (vary `ticks_per_second` —
`dt` feeds the logistic; vary fire params) self-matches at `tol=0.0`. Erik's feel-check on a firestorm.

---

## 4. Substep / CFL cliffs

**Fire has NO substep loop today** — `fire.step` runs **once per tick at full `sim_time`** (`physics_engine.cpp`
calls it once in `step_tail`; engine/06 §"Implementation status" confirms "once per tick at full sim_time").
The logistic `I += dt·(grow−die)` is an explicit forward-Euler step with no CFL substepping (the gains
`k_grow=4`, `k_die=2` × `dt≈0.042` keep it well inside stability). **So there is no integer cliff to make
fixed-point in S3** — unlike S1's water `n`, S2's `n_wave`/`n_smoke`. If a future tuning pass ever adds a
fire substep count (e.g. for a much larger `k_grow`), it would use `ceil_div` (`fixed_point.h:189`) over a
config-constant CFL → the easy cliff. **Flag for impl:** confirm no hidden `dt`-derived integer count
sneaks in; today there is none.

---

## 5. Determinism hazards (ranked) + handling

1. **The per-cell `sqrt` (`W`, `:86`) — the FIRST per-cell transcendental of the whole arc, the only
   genuinely new determinism problem.** A host `std::sqrt` per cell is NOT in the deterministic-path
   contract (master plan §5.1 / §5.3: no host `Math.Sqrt`, no third-party float lib). **Handling:**
   `sqrt_q16` = fixed-32-iteration integer `isqrt64` of the Q.32 radicand → exact floor-√, branch-identical,
   identical CPU↔`__device__`, no LUT/poly/libm (§2.4). The single most important new primitive in S3.
2. **The 6-factor logistic chained truncation (M2) — order-dependent narrowing.** Each `mul_q16` truncates;
   the product `k_grow·avail·hot·I·(1−I)·(1+k_wind_fan·W)` gives different last bits under different
   association. **Handling:** carry the chain in a wide int64 (Q16.48), **narrow once** at the `dt·(grow−die)`
   write, **pin the multiply tree order** explicitly in the code (§3 S3b). Deterministic by construction
   once the order is fixed; the gate asserts the burning↔extinguishing flip tick is bit-identical.
3. **The discrete burning↔extinguishing flip + the burn-through list — control-flow outputs.** `I_next <
   I_min → 0` and `wall_hp <= 0 → destroyed` are discrete events that drive synced state (the destroyed-tile
   list → `destroy_wall`, the renderer's fire on/off). A 1-LSB difference at the threshold flips the event on
   a different tick → desync. **Handling:** integer compares on integer fields → bit-identical by
   construction; gate on the digest of `fire` + the destroyed list across two runs of an ignite→extinguish
   trajectory.
4. **The Python ignition O2 float mean (`combat.py:330-344`) — a Python float bridge deciding who ignites.**
   **Handling:** integer neighbour-sum + rounded mean on int32 atmosphere, bit-matched to the C++ fire `P`
   gate (§2.2, S3a). The threshold quantizes once. (Python *int* math is cross-machine-exact per Q2; the
   hazard is specifically the *float dequantize* on the bridge, which S3a removes.)
5. **The plume + smoke-emission DEPOSITS — truncation bias accumulation.** These are non-conserved deposits;
   plain `mul_q16` truncation toward −∞ would accumulate a small DC bias into atmosphere/smoke over a long
   firestorm (the S2a/S2c lesson: ROUND-TO-NEAREST for unbiased deposits, NOT for cancelling flux pairs).
   **Handling:** round-half add on the deposit narrow (§3 S3b); gate on mean-signed-error ≈ 0.
6. **No RNG, no global reduction in fire.** The fire step has **no RNG** (engine/06 §5 confirms: no random
   ignition/spread — spread is radiation→heat→ignition, deterministic; the only RNG in the file's
   neighbourhood is `combat.py`'s bullet cone, already seeded, out of scope). The only reductions are the
   `max_fire` early-exit (order-free integer max) and the `P` neighbour-mean (tiny local sum, order-free).
   **No hazard** beyond making the `max_fire` compare integer.
7. **Overflow watch (the new-burden, master plan §8.1).** The Q16.48 logistic intermediate and the int64
   Q.32 wind radicand must not overflow int64. `wx²+wy²`: `|wind|` is gradient-scale O(1), spiking under a
   shockwave — even a generous `|wind|≈100` gives `wx²+wy²≈2e4` real → Q.32 ≈ `2e4·2^32 ≈ 8.6e13 < 2^63`,
   ~17 bits of headroom; fine. The logistic factors are all `[0,1]`-ish except `(1+k_wind_fan·W)` — bounded
   by the same `W`. **Handling:** declared-bounded ranges + a debug assert at every int64→int32 narrow,
   driven by the shockwave-fanned firestorm stress scenario (master plan §6.2.5).

---

## 6. Gating (the acceptance contract)

Mirrors S2 §4, fire-specialized:

- **P1 — within-config bit-identity** (`tol=0.0`, run twice, bit-identical via `field_ab_harness.py`; `fire`
  is already in `SIM_FIELDS` `:74-75`, now digested as int32) **AND cross-config self-consistency** (vary
  `ticks_per_second` — `dt` feeds the logistic; vary the `[physics.fire]` params). Each sub-step
  self-matches.
- **No P2 conserve-to-LSB for fire** — fire is **not conserved** (logistic source/sink). Its writes into
  *conserved* fields are the gated thing: the **plume deposit into atmosphere is unbiased** (mean signed
  error ≈ 0, §5.5) and **atmosphere bulk mass P2 still holds** (the deposit is the intended source
  exception, like the sponge). Smoke is SLint (non-conserved, accepted from S2b) — the emission deposit is
  gated on unbiasedness, not conservation.
- **Discrete-output determinism** (the fire-specific gate): the **burning↔extinguishing flip tick** and the
  **burn-through destroyed-tile list** are bit-identical run-to-run across an ignite→firestorm→starve
  trajectory. This is fire's analogue of "converges" — the discrete events must land on the same tick.
- **Feel-regression vs the float golden** (S3b changes behaviour — integer truncation ≠ IEEE): scripted
  firestorm scenario (ignite, wind-fan, blow-out, extinguish) under (i) the float golden and (ii) the
  integer build, compared with a physical tolerance on `fire` (`atol≈2e-4`, max-cell + L2) **and mean signed
  error ≈ 0**; render N frames to PNG both sides, SSIM/per-pixel diff, **committed artifact Erik reviews**
  (master plan §6.4). The ignition-tick and extinguish-tick must match within a tick.
- **The ignition O2 invariant test** (S3a): the C++ fire `P` gate and the Python `apply_temperature_ignition`
  O2 gate return the identical boolean on a shared cell across a sweep of atmosphere values (they MUST agree,
  `combat.py:276-281`).
- **Goldens regenerated + version-bumped in the same commit** (fire dtype flip + the integer behaviour
  change); the Q-format version tag records **fire = int16 (Q1.15) GPU width** (Q5, frozen now) even though
  CPU storage is int32.
- **Full suite green + both `--auto` exit 0.**

---

## 7. OPEN QUESTIONS for Erik (the decisions — distinct from the mechanical conversions)

> These are the genuine physics/design/feel choices where the integer change *could* alter behaviour in a
> way a human must bless. The mechanical conversions (every divide → load-time reciprocal, the smoke/plume
> deposits → round-to-nearest, the bridge deletion) are NOT here — they are settled by the established arc
> patterns. Recommendations given for each.

- **Q1 — `sqrt_q16` rounding: floor-isqrt (exact, simplest) vs round-to-nearest isqrt.** `W = |wind|` feeds
  the gentle, non-conserved wind-fan/strip terms. Floor truncates `W` by up to ~1.5e-5 (1 LSB),
  deterministically. **Recommend FLOOR-isqrt** — exact by construction, no LUT/poly, no committed data file,
  identical CPU↔GPU, and the 1-LSB bias on a tuning term is perceptually invisible (and deterministic →
  behaviour, not desync). Round-to-nearest is one extra compare if a feel-check ever wants it; it is not
  needed. *(This is the only genuinely new transcendental decision of the whole arc — worth an explicit
  yes.)*

- **Q2 — `wall_hp` format: int32 Q16.16 vs keep it as a small-integer count.** `wall_hp` is HP (wood=30,
  integer-ish), used only as `F = clamp01(wall_hp/fuel_ref)` (a normalized [0,1] fuel fraction) and depleted
  by `wall_damage·dt·I`. It does NOT need fractional precision for gameplay, but the depletion `wall_damage·
  dt·I` IS fractional (`0.4·0.042·I` per tick ≪ 1). **Recommend int32 Q16.16** (fractional depletion
  accumulates correctly; `F` is a clean Q16.16 normalize). The alternative (integer HP + a separate
  fractional-damage accumulator) is more code for no benefit. Flagged because it is a storage decision a
  human should ratify (it touches the wall/material system, which has other readers — confirm no other
  consumer of `wall_hp` expects a plain integer).

- **Q3 — the `max_fire < 0.001f` early-exit threshold + `I_min = 0.02` snap-extinguish: confirm the Q16.16
  quantized thresholds are the intended values.** These become `quantize(0.001)` and `quantize(0.02)`
  integer compares. Quantization rounds them to the nearest LSB (0.001 → 65 counts, 0.02 → 1311 counts) —
  effectively exact. **Recommend: quantize as-is, no behaviour change.** Trivial, but it is a threshold that
  gates a discrete event (the extinguish flip), so worth a confirm that 0.001/0.02 are not load-bearing to
  more precision than Q16.16 gives (they are not).

- **Q4 — the plume + smoke-emission deposits: round-to-nearest (recommended) vs the truncating `mul_q16`.**
  These are non-conserved deposits into atmosphere/smoke. The arc's S2a/S2c lesson says ROUND-TO-NEAREST for
  unbiased deposits (truncation accumulates a DC bias over a long firestorm). **Recommend round-to-nearest**
  (a round-half add on the deposit narrow). This is a (tiny) behaviour choice — the float build has no such
  quantization at all — so the feel-gated A/B must confirm a firestorm's smoke/pressure output is
  indistinguishable. Calling it out so Erik knows the integer build makes a *rounding* choice the float
  build never had to.

- **Q5 — the Python combat HP fence (Q2-LOCKED at the master-plan level): confirm S3 does NOT migrate
  `apply_environmental_damage`/`apply_blast_damage`, only adds the harness digest of unit HP/events.** S3 is
  where the fire→heat→damage loop closes, so the float-HP leak first becomes *exercisable* (a fire can now
  deterministically heat a unit, and the float damage math is the next link). The master plan §9 Q2 keeps
  this Python-float for now (cross-machine-exact for scalar math) with the harness extension owed.
  **Recommend: hold the fence, but land the unit-state digest extension AS PART OF S3's gate** (so the fire
  determinism we just built is actually watched end-to-end through to the kill event, not silently leaked at
  the HP step). This is the one place S3 touches the Q2 boundary, and it is additive (a digest), not a
  migration. Confirm the priority: digest now, C++ HP migration later.

---

## 8. Risks

### 8.1 The `sqrt_q16` correctness (the new primitive)
A wrong isqrt is a silent desync. **Mitigation:** the property unit test (`sqrt_q16(x)² ≤ x <
(sqrt_q16(x)+1)²` over sampled int64 in-range), the same fixed-iteration discipline as `tan_poly`, and the
reduction-permutation P1 test (a non-deterministic sqrt fails it instantly on the CPU, before any GPU).

### 8.2 The two fire-write paths drifting apart (C++ `step` vs Python `apply_temperature_ignition`)
They must agree on the O2 predicate (the design-note invariant) and on the Q16.16 fire representation. If
S3a's Python integer O2 mean and S3b's C++ `mean_round` diverge by a rounding rule, a tile ignites into a
state the fire kills (or vice versa) — a behaviour bug, deterministic but wrong. **Mitigation:** the shared
`mean_round` contract (§2.2) + the explicit cross-path equality test (§6).

### 8.3 The logistic-chain overflow under a shockwave-fanned firestorm
`(1+k_wind_fan·W)` with a large transient `W` is the one unbounded-ish factor. **Mitigation:** the declared
int64-intermediate bound (§5.7), the debug assert at every narrow, and a stress scenario that drives a
grenade shockwave through a blaze (the master plan's `destroy_wall(8,0)` + a fire seed) so the worst-case `W`
is actually exercised.

### 8.4 The temperature TU signature change (S3c)
Converting `temperature_solver`'s `atmosphere` arg from `float*` to `int32_t*` touches a shipped, green TU.
**Mitigation:** it is the LAST float input to temperature (it becomes fully integer, consistent with its
already-integer heat/temperature fields); it is a mechanical signature + one threshold-compare change; the
per-cell A/B harness gates it at `tol=0.0`; the `wave_p_f_` buffer is untouched (separate water bridge).

### 8.5 Fire-dtype flip ripples into Python readers
`gmap.fire` going int32 means the renderer, `cast_fire_heat`, and any test reading `gmap.fire` need a
dequantize boundary. **Mitigation:** audit all `gmap.fire` readers in S3a (it is a small set — fire is
young); the harness already handles the smoke int32 view the same way (`field_ab_harness.py`), so the
pattern is established.

---

*Companion docs: `docs/fixed_point_migration_plan.md` (the master plan + §9 LOCKED resolutions Q1-Q9),
`docs/s2_fixed_point_plan.md` + `docs/s2_atmosphere_group_map.md` (the structure + rigor mirrored here),
`cpp/src/fixed_point.h` (the shipped toolkit — `reciprocal_q16`, `mean_sum`/`mean_round`, `mul_wide`/`narrow`,
`scale_mag`, `ceil_div`, `tan_poly`; S3 ADDS `sqrt_q16`), `cpp/src/temperature_solver.cpp` (the int32 template
fire reads), `cpp/src/smoke_dynamics.cpp` (the int32 SLint fire writes), `docs/architecture/engine/06_temperature_and_fire.md`
(the canon fire chapter — §5 the logistic, §"Implementation status"). The fire bridge to collapse:
`cpp/src/physics_engine.cpp:89-133` + the scratch buffers `physics_engine.h:54-56`. The Python ignition twin:
`src/simulation/combat.py:253-352`. Shipped `[physics.fire]` params + `ignition_seed`/`o2_threshold`:
`config.toml:70-105`. All line numbers verified at HEAD `9e0e456` (2026-06-26); re-verify before impl.*
