Proposal from the temperature-design research pass (2026-06-07), completing engine/06 §1–§4. **Status: proposal for discussion — nothing here is canon or implemented until agreed.**
Build-FIRST: this is the temperature substrate to land before the fire mechanic; once agreed it folds into engine/06.

# Temperature System — Complete Design Proposal (engine/06 §1–§7)

> **Status:** PROPOSAL for review (discuss-before-implement). Folds into `docs/architecture/engine/06_temperature_and_fire.md`. Builds **on** the existing §1–§4: does not restate the locked decisions (radiation-only heat into the `heat` buffer, temperature on solids only, κ=0 air no-op, one relaxation pass/tick, Q16.16 lockstep determinism, fire-as-ray-source). It **adds** the missing pieces (heat→temperature conversion, ambient cooling, the precise unit-flux model, temp→pressure) and **refines** §2 where the research found a cleaner choice.
>
> **Build-FIRST:** this is the temperature *substrate* to land before the fire mechanic. Fire is the consumer that proves it.

---

## 1. Heat → temperature conversion (new — fills the §1↔§2 gap)

The `heat` buffer is a **per-tick deposit** (Q16.16, filled by the read-only ray pass, order-independent saturating-add). It currently has no consumer and is cleared unused. This section turns deposited heat into persistent `temperature` on solids, via a new per-material **thermal mass**.

### 1.1 New material property: `thermal_mass`

Add one column to `MaterialTable` / `[materials]`: **`thermal_mass`** — the heat-units needed to raise one tile by one temperature unit (the analogue of volumetric heat capacity ρ·c·V, collapsed per-tile since all tiles share one volume).

Authored as **powers of two** so the conversion is an exact right shift. The values are anchored to real volumetric heat capacity ρ·c (steel ≈3.8, glass ≈2.1, softwood ≈0.7 MJ/m³K → steel needs ~5× wood's energy/degree) but the spread is **exaggerated to 8×** so the gameplay reads clearly: metal soaks energy and glows as a conducting plate; wood spikes hot at the impact point and ignites.

| Material | real ρ·c (MJ/m³K) | `thermal_mass` | stored `heat_inv_shift` = log₂ | behaviour |
|----------|------------------:|---------------:|-------------------------------:|-----------|
| Hull  | ~3.8 (steel) | 64 | `>> 6` | metal: many joules/degree — heats slowly per ray, §2 spreads it fast |
| Steel | ~3.8         | 64 | `>> 6` | as hull |
| Glass | ~2.1         | 32 | `>> 5` | half metal's mass — heats ~2× faster locally |
| Door  | ~0.7 (wood)  |  8 | `>> 3` | light wood — heats fast, ignites readily |
| Wood  | ~0.7         |  8 | `>> 3` | heats fast and **locally** (low κ keeps it where it landed) |
| Air   | — (κ=0)      |  1 | `>> 0` | irrelevant: temperature is discarded on air every tick; value non-zero only to avoid a divide-guard |

`thermal_mass` is the *only* table column that sits on a divide, so it is the only one constrained to powers of two. `conductivity`, `ignition_temp`, wall limits stay arbitrary — they never divide. At load, `heat_inv_shift = log₂(thermal_mass)` is precomputed into a per-tile cache **parallel to the existing `conductivity` cache**, and patched by the same `on_tile_changed` seam when a tile changes.

### 1.2 The exact conversion update (Q16.16, division-free)

Both `heat` and `temperature` are **Q16.16 int32**, sharing one scale (`TEMP_SCALE = HEAT_SCALE = 65536`), so `ignition_temp`/wall-limits — quantized once at load — compare directly against both with no rescale. Per solid tile, run **before** relaxation:

```
shift = heat_inv_shift[y,x]                 # precomputed log2(thermal_mass), 0..6
gain  = heat[y,x] >> shift                   # Q16.16 / 2^shift  ==  J / thermal_mass, still Q16.16
temperature[y,x] = sat_add_i32(temperature[y,x], gain)   # reuse the shipped heat_saturating_add
```

Fixed-point guarantees:

- **`>>` on a Q16.16 value divides the represented quantity by 2^shift while staying Q16.16** — the binary point does not move. One heat-unit into wood (`>>3`) = 1/8°; into steel (`>>6`) = 1/64°.
- **Reuse the shipped `heat_saturating_add` / `sat_add_i32`** (`cpp/src/raycaster.h`): temperature pins at `INT32_MAX` under a firestorm instead of wrapping cold — identical guarantee to the deposit side.
- **Arithmetic right shift on non-negative int32 is portable and bit-identical** across machines/compilers (`heat` is a saturating accumulator of non-negative deposits, so the sign-edge case never arises). No float, no division → the Level-2 cross-machine lockstep guarantee holds; the float-temperature fallback noted in §3 is **not needed**.
- **Non-2ⁿ escape hatch (not used by the shipped table):** store `heat_inv_q16 = round(65536 / thermal_mass)` at load; `gain = (int64(heat) * heat_inv_q16) >> 16` (int64 mulhi, deterministic). Kept only if a future material wants a non-power-of-two feel.

### 1.3 Heat-buffer clear & per-tick ordering for conversion

The clear must move so conversion reads `heat` **before** it is wiped — but the `heat` buffer also feeds **unit damage** (§4) and the **render glow** (`smoke_glow`/`heat`). So the clear stays at **end of tick, after every `heat` consumer** (conversion, unit-damage, render-sample), not immediately after conversion. Net change from today: the end-of-tick clear simply moves to *follow* the new consumers; convert/cool/damage steps are inserted *before* it. Single-buffer, cleared at end of tick is the cheap deterministic choice; double-buffer (`heat_prev` for render/damage) only if render timing ever demands it. Full order in §6.

---

## 2. Conduction relaxation — made precise (refines existing §2)

This **confirms and completes** the locked §2 (one relaxation pass/tick, power-of-two rates, κ=0 no-op). The research refined three things: **4-neighbour** (von Neumann) over 8, **harmonic-mean face conductivity** resolved into a load-time table, and the **damped-Jacobi stability argument** (not CFL).

### 2.1 Field & format

`temperature` — dense full-grid **Q16.16 int32**, same format/scale as `heat`. Double-buffered (`temp` → `temp_new`, swap) for an order-independent gather.

### 2.2 The exact per-tick update formula

Relaxation toward a conductivity-weighted neighbour blend. For tile `i`, neighbours `n ∈ 4-nbr(i)`:

$$T_i' = T_i + \sum_{n} r_{i,n}\,(T_n - T_i), \qquad r_{i,n} \in \{2^{-2}, 2^{-3}, \dots\} \cup \{0\}$$

In Q16.16 integer code (gather stencil, fixed N,S,E,W order, 64-bit accumulator):

```c
int64_t acc = 0;
for (dir in {N,S,E,W}) {
    int s = face_shift[i][dir];           // precomputed; NO_FACE = grid edge or κ=0 either side
    if (s == NO_FACE) continue;
    int32_t dT = temp[n] - temp[i];       // signed Q16.16 difference
    acc += (int64_t)dT >> s;               // arithmetic shift = ÷2^s
}
temp_new[i] = (int32_t)((int64_t)temp[i] + acc);   // single write per tile
```

Two structural rules that keep it exact and stable:
- **Gather, not scatter** — each tile writes only itself, from a frozen previous-tick buffer. No atomics, order-independent by construction (temperature determinism rests on *exact shifts*, not on add-associativity the way `heat` does).
- **Shift the *difference*, not the neighbour** — `(T_n − T_i) >> s`, so equal neighbours produce *exactly zero* change (no drift to a quantization floor) and flux is conservative-shaped.

### 2.3 4 neighbours, not 8 (refinement)

Use **von Neumann (4)**. A diagonal "face" is a corner contact with no clean area; 8 equal power-of-two terms conduct √2 too fast diagonally and spread hot spots into diamonds. 4-neighbour is the standard isotropic-enough stencil, makes the stability bound trivial (§2.6), and costs 4 shifts + 4 compares per solid tile (air skipped). Fire *spread* already gets its reach from rays (§5); conduction is the slow along-the-metal channel and needs no diagonal.

### 2.4 conductivity → power-of-two self-rate

Physical conduction speed scales with thermal diffusivity, which is **logarithmic** in character (steel ≈62× wood in real diffusivity; the `conductivity` column already compresses 50 vs 0.15 ≈ 333×). Map κ to a shift via a base-2 log bucket, **computed once at load, never per tick**:

```
shift(κ) = clamp( SHIFT_MIN, round( SHIFT_AT_REF − log2(κ / KAPPA_REF) ), NO_FACE )
rate     = 2^(−shift)
```

With `KAPPA_REF = 50` (hull) and `SHIFT_AT_REF = 2` (rate ¼ — fastest stable on a 4-nbr grid, §2.6):

| Material | κ | log₂(κ/50) | shift | rate | feel |
|----------|----:|----------:|------:|-----:|------|
| Hull  | 50.0 |  0.00 | **2** | 1/4    | races along metal |
| Steel | 45.0 | −0.15 | **2** | 1/4    | same bucket as hull (correct: both "metal") |
| Glass | 1.0  | −5.6  | **8** | 1/256  | middling-slow |
| Door  | 0.3  | −7.4  | **9** | 1/512  | wood-like |
| Wood  | 0.15 | −8.4  | **10**| 1/1024 | slow → stays hot locally → ignites |
| Air   | 0.0  | —     | NO_FACE | 0   | **no conduction** |

`log2` appears **only here, at load, in float**, then frozen to integer shifts. Coarse 2× buckets are deliberate — plenty of gameplay resolution, division-free at runtime.

### 2.5 Face conductivity between two materials — harmonic mean (refinement)

Two materials in series add their *resistances*, so face conductance is the **harmonic mean**, not the arithmetic mean. This correctly makes a wood/metal face conduct at ~the wood rate (a face touching an insulator nearly closes); arithmetic mean would leak heat into insulators too fast. Harmonic mean needs a division — so **resolve it into a precomputed `N×N` face-shift table at load**:

```
face_shift_table[a][b] = round( -log2( harmonic_mean(κ[a], κ[b]) / KAPPA_REF ) )   # one float div, LOAD ONLY
                         clamped to [SHIFT_MIN, ...],  NO_FACE if either κ == 0
where harmonic_mean(a,b) = 2ab/(a+b)
```

6×6 table today. Runtime only ever indexes `face_shift_table[material[i]][material[n]]` and shifts. Two properties fall out:
- **κ=0 ⇒ NO_FACE on every face it touches** — air conducts to nothing and no solid conducts into air. The κ=0 no-op is enforced *structurally* in the table, not by a runtime value-branch.
- **Symmetric table ⇒ symmetric flux** — `face(a,b) == face(b,a)`, so the flux at `i` is the negation of the flux at `n` (up to the LSB floor note in §2.7).

The GameMap bakes a per-tile `face_shift[y][x][dir]` cache from the material grid and patches it in `on_tile_changed` — the **same seam** already used for occlusion/conductivity — so a breached wall updates its faces the instant the tile changes.

### 2.6 Unconditional stability — the correct argument (refinement)

This is **not** an explicit heat equation, so **CFL does not apply** (the old "~17 substeps" worry is moot). It is a **damped-Jacobi / convex relaxation**. Rewrite the update as a weighted average:

$$T_i' = \Big(1 - \sum_n r_{i,n}\Big)T_i + \sum_n r_{i,n} T_n$$

If all weights are non-negative and **sum to ≤ 1**, `T_i'` is a convex combination of `{T_i, T_n}`, so the **discrete maximum principle** holds: `min_j T_j ≤ T_i' ≤ max_j T_j`. No new extremum is ever created; global max is non-increasing, global min non-decreasing → the field is bounded by its own data **for all time, any rates** satisfying the weight condition → **unconditionally stable**, independent of tick size. The iteration matrix is row-stochastic (spectral radius ≤ 1); there is no amplification factor to violate.

**The one condition:** `Σ_n r_{i,n} ≤ 1` per tile. With `SHIFT_MIN = 2` (max face rate ¼) and 4 neighbours, the worst case is `4 × ¼ = 1` exactly — binding and **safe** (self-weight ≥ 0). This is *why* `SHIFT_MIN` is pinned at 2: shift 1 (rate ½) would let four metal neighbours sum to 2, break convexity, and oscillate. (8-neighbour would force `SHIFT_MIN ≥ 3` — another reason 4 is clean.)

**κ=0 air no-op — proven:** air has every face = `NO_FACE`, so `Σ_n r = 0` ⇒ `T_i' = T_i`. Combined with conversion skipped on κ=0 (§1.2 runs on solids only), an air tile that starts at 0 stays bit-exactly 0 forever — never source, never sink.

### 2.7 Cross-machine bit-exactness

Runtime update is **pure signed addition + arithmetic right shift** — fully specified by the ISA, identical on every machine/compiler, no float, no FMA, no reassociation. Every `log2`/harmonic-mean/division happens **once at load in float**, rounds to an integer shift, freezes into the tables (which can be precomputed and committed as data if even load-time float rounding is a concern). Gather + double-buffer ⇒ order-independent. The only rounding is arithmetic right shift rounding **toward −∞** uniformly — a deterministic *bias*, not *drift* (same on all machines, never desyncs). It makes flux not perfectly antisymmetric across a face when `(T_n−T_i)` is odd (≤1 LSB = 2⁻¹⁶° per face per tick), which is immaterial — temperature is a threshold signal, not a conserved budget. *(If exact antisymmetry is ever wanted: iterate faces not tiles, compute each flux once, apply `+flux`/`−flux`. Recommend shipping the gather version; the bias is negligible.)*

---

## 3. Ambient cooling — NEW (fills the gap; this is what lets fires burn out)

Without cooling, every wall latches hot forever and thresholds become irreversible. Cooling is the **negative term in the fire's energy balance**: once a fire's per-tick `heat` deposit no longer exceeds its per-tick cooling loss (fuel/`wall_hp` depleted or O₂ proxy drops), the tile relaxes below `ignition_temp` and the fire dies. Newtonian (linear) decay toward ambient, same shift discipline as conduction.

### 3.1 T_ambient = 0 (store ΔT above a reference)

Store temperature as **degrees above ambient**, so **`T_ambient = 0`** in fixed-point. Then cooling is `delta = T >> COOL_SHIFT` with no subtraction, "cold" is the natural empty state of a freshly-allocated int32 field, and `ignition_temp` (wood 300) / wall limits are stored as offsets above the reference, quantized once at load. (For a literal °C UI readout, add 20°C at *display* time only — never in sim.)

### 3.2 The per-tick cooling update

Continuous law `dT/dt = −k·(T − T_ambient)`, discretized with a power-of-two rate (shift+add, no division):

```
T -= T >> shift          # since T_ambient = 0; equivalently T += (0 - T) >> shift
```

This is the same unconditionally-stable relaxation form as §2, with a fixed target (0) instead of a neighbour blend — `T` relaxes toward ambient and never crosses it for a single tile (the shifted magnitude is always ≤ the difference).

- **Arithmetic right shift on the signed value** (rounds toward −∞, deterministic, identical cross-machine; pre-C++20, pin with `x<0 ? -((-x)>>s) : (x>>s)`). Cold tiles below ambient warming back is symmetric in practice and invisible at gameplay scale.
- **Residual dead-band is desirable:** the last `(1<<shift)−1` counts above ambient never decay (shift yields 0) — an exact, jitter-free resting state at `T_ambient`. **Do not** add a "+1 if nonzero" nudge; it would break the clean fixed point and reintroduce a division-like asymmetry.

### 3.3 Rate, and vacuum-exposure faster cooling (recommended)

Per-tick fractional decay is `2^(−COOL_SHIFT)`:

| `COOL_SHIFT` | per-tick decay | ~e-fold (ticks) | feel |
|---|---|---|---|
| 6 | ≈1.6% | ~64 | very slow |
| **5** | **≈3.1%** | **~32** | **recommended interior default** |
| 4 | ≈6.25% | ~16 | fast |
| 3 | =12.5% | ~8 | aggressive — use for vacuum-exposed |

**Recommendation: vacuum-exposed tiles cool faster.** Physically, a sealed room loses heat by convection (Newton's law literally), while a space-facing plate at fire/laser temperatures sheds fast by radiation — and "breached wall cools quickly, interior wall stays hot and keeps re-igniting wood" is a strong, readable gameplay signal. Implement as a **1-bit per-tile choice between two power-of-two shifts**, from a 4-neighbour read of the *existing* atmosphere/vacuum field (no new field, no new buffer):

```
exposed = any 4-neighbour is vacuum-flagged OR atmosphere[nbr] < o2_vacuum_thresh
shift   = exposed ? COOL_SHIFT_VACUUM : COOL_SHIFT      # 3 : 5  → 4x faster
T      -= T >> shift
```

Reuse the **same neighbour gather** the conduction pass already does (the four cells are already in hand) and tie the vacuum flag to the one `destroy_wall`/`on_tile_changed` already maintains (ch.02) — so the instant a hull breaches, newly space-facing walls flip to the fast shift through the existing seam, no special-casing.

> **Why linear (Newton), not Stefan-Boltzmann T⁴:** real radiative loss scales as T⁴, which needs a multiply/LUT and breaks the shift-only contract. Newton's linear law is the empirically-correct form for *convective* loss and holds well for moderate ΔT; we absorb the "hotter radiates faster" effect into the rate constant, not the exponent — consistent with the chapter's "faked, unconditionally stable, not thermodynamically accurate" stance.

### 3.4 Global, not per-material (recommendation)

Make the cooling rate **global** (the two shifts above), **not** a per-material column:
1. **Conductivity already encodes "metal sheds differently"** — high-κ metal spreads heat across the whole connected plate, lowering local T so global cooling acts on a larger reservoir; low-κ wood holds heat locally. A per-material rate would *double-count* this.
2. **The legible variation is environmental** (exposed-to-space vs interior = the §3.3 vacuum flag, dynamic and gameplay-driven), not material.
3. **Tuning surface:** two global shifts are far easier to balance against the fire deposit than six per-material rates.

If a future "insulated bulkhead" ever needs an exception, add it as a **larger shift selected by a 1-bit material flag** (insulated yes/no), keeping it to a small set of power-of-two shifts — never a continuous per-material rate. Document such a column as **deferred** (mirroring `emissivity` in ch.03).

### 3.5 Ordering

Cooling is the **last thermal pass**, after relaxation: conduction must redistribute *this tick's* fresh deposit across the metal **before** any of it is shed (else a hull hit cools at the impact point before racing down the plate — losing the §6 "heat travels along metal" payoff), and cooling must run **before** consumers so thresholds test the *net* post-loss temperature (a space-exposed wall under a weak beam that received heat but also shed it is correctly judged *not* to ignite — the burn-out mechanism). Full order in §5/§6.

---

## 4. Units & heat — made precise (fills the §4 "unit damage" gap)

A unit takes heat damage from **incident radiant flux**, which is exactly the energy the ray engine already deposited at the unit's tiles. No new occlusion: units stamp themselves as full blockers before the ray pass (`stamp_units`), so rays terminate **on** their leading tiles and the `heat` buffer *is* the correctly-occluded, distance-attenuated incident-flux field. This is the unit-side mirror of "the kernel never writes the unit; the unit samples the buffer after the pass."

### 4.1 What a unit samples — incident flux Φ

Per living unit, **after** the ray pass fills `heat`, **before** the end-of-tick heat clear:

**Direct radiant term — `max` over the footprint** (footprint-size invariant; matches the physics that the hottest tile on the body is the exposure that matters; reads off the already-occluded field so a unit half behind a corner has its shadowed tiles ~0 and `max` picks the exposed side that burns):

```
Φ_rad = max( heat[ty, tx] for (tx, ty) in unit.occupied_tiles() ) / HEAT_SCALE
```

**Adjacent-contact term — optional, recommend deferring** (a short gather over the 1-tile-dilated footprint border, for the edge case of a unit pressed against a flame on a tick the ray missed its exact tiles):

```
Φ_contact = max over border of:  k_fire_contact · fire[n]
                                 k_solid_contact · max(temp[n] − T_touch_ref, 0)   # gated on temperature field
Φ = max(Φ_rad, Φ_contact)        # max, not sum: same physical quantity, summing double-counts
```

The `temp[n]` half activates once temperature ships; the `fire[n]` half once fire casts rays. **Recommendation: ship `Φ_rad`-only first** (a single buffer read covers beams and radiated fire completely); add `Φ_contact` only if playtesting shows ray-miss ticks letting a flame-hugging unit off — build the consumer, prove the need before adding mechanism.

### 4.2 Flux → damage-per-tick

```
Φ_abs  = Φ · unit_absorption · (1 − unit_reflectivity)          # shipped [combat] consts: 0.85·0.90 = 0.765
T_felt = T_ambient_ref + k_flux_to_temp · Φ_abs                  # radiation drives felt temp; NO air-temp field
over   = T_felt − temperature_max                                # EnvironmentProfile band
if over <= 0: no heat damage this tick
dmg    = environmental_damage_rate · (1 + k_over · over) · dt_tick   # reuse the env-damage channel
if u.is_zombie: dmg *= zombie.fire_damage_multiplier            # shipped 4.0 — THIS is its intended site
u.current_hp -= dmg
```

- `dt_tick = 1/ticks_per_second` keeps it tick-rate independent; `k_over` ramps damage with over-temperature (`k_over = 0` → flat "outside band = constant damage"; lean small positive so a furnace is lethal, a warm room survivable).
- A future reflective vac-suit becomes a modifier on the *effective* `unit_absorption`/`unit_reflectivity` (base/effective stat pattern) — suited marine takes less, for free.
- **Damage type:** heat deaths set `source="heat"` on hit/killed events and do **not** set `killed_by_zombie` (like blast/bullet — only melee converts; a burned corpse converting would be wrong).

### 4.3 Determinism

Pure **gather + serial apply**, inheriting the via-physics guarantees with no new surface:
- **The buffer is already deterministic** — `heat` is order-independent integer saturating-add. The unit step reads it **non-destructively** and must run **before** the heat clear. No atomic writes into the unit.
- **Sampling is an order-independent gather** — `max` over footprint/border has no inter-unit dependency; overlapping units both read a hot tile, neither consumes it.
- **Apply is a serial CPU loop in fixed unit order** — iterate `self.units` in stored order, skip dead, subtract `dmg`, flip `alive`/emit events on `hp ≤ 0` — the same discipline shipped for `apply_blast_damage`.
- **Fixed-point:** for the first single-machine cut, float `dmg` is acceptable (the heat *read* is already integer); cross-machine bit-exactness folds in with the engine-wide fixed-point migration (same fallback as §3: float damage + integer heat-deposit = single-machine determinism).

New serial step `apply_environmental_damage(units, gmap)` slots into `Simulation.step()` after physics fills `heat`, before the recorder snapshot (so the recorder captures post-damage state) and before the heat clear. Its existence is precisely what makes clearing `heat` correct — a consumer finally reads it before reset.

---

## 5. Temperature → pressure — recommendation: keep it FIRE-SPECIFIC

**Do not add a general temperature→pressure coupling or a general air-temperature field.** The fire's own-tile pressure deposit is the **whole** mechanism.

### 5.1 Why (and why 2D is decisive)

Real compartment-fire convection has two drivers: (1) **inhibited thermal expansion** → local overpressure → in-plane outflow, and (2) **buoyancy/plume** → vertical rise, ceiling jet, hot upper layer. **The game is 2D top-down, so driver (2) is entirely out of plane** (ch.05 §6.1 already states no buoyancy term). The only convection driver that survives projection to the floor plane is (1) — local overpressure → in-plane outflow — which is **exactly** what the locked fire O₂-fix already does by depositing pressure on the fire tile. The fire-specific deposit captures 100% of the convective effect a top-down view can show; a general air-temperature field would model only the invisible part.

A general field is the wrong layer:
1. **It re-introduces the field §1 forbids, for nothing** — "hot solid heats adjacent air → air temperature drives pressure" *is* an advecting air-temperature field by another name, with a single consumer (pressure) the fire deposit already serves. Ignition/damage read the `heat` buffer (radiation), not air temperature.
2. **The atmosphere solver already gives the convection for free** — the deposit goes into `atmosphere`; IMEX implicit diffusion (ch.04 §2.4) spreads it and wind `−∇(atmosphere+wave_p)` carries smoke out. This is the proven `apply_explosion` direct-deposit pattern, just small and continuous.
3. **Determinism cost, zero return** — a second dense field would need its own diffusion/advection pass and its own fixed-point validation; the fire deposit rides the *existing* field and solver, no new determinism surface.
4. **Runaway pressure is already solved emergently** — the ch.04 §2.7 over-pressure relief valve (`find_burst_walls`, per-material `burst_threshold`): a sealed-room fire over-pressurises → weak bulkheads burst → room vents → fire gets fresh O₂ proxy through the breach → firestorm propagates. Real ventilation-driven spread, falling out of two shipped systems with no new code.

### 5.2 The rule (the locked fire O₂-fix, stated precisely)

```
# fire tick, after intensity update
atmosphere[y,x] += fire_pressure_gain * fire_intensity[y,x] * dt    # FieldEdit deposit, own tile
o2_proxy = mean(atmosphere[air-side neighbours])                    # read-only; does NOT remove atmosphere
```

### 5.3 Reconciliation with §1's no-air-temperature-field principle

§1 forbids a *diffusing temperature field on air*. The fire deposit stores **no temperature on air** — it writes a pressure *impulse* into the existing `atmosphere` field, then forgets it. Air still holds `κ=0` and no temperature. The thermal→kinetic conversion happens **once, at the source tile, at deposit time** (intensity → pressure), never as a stored advecting quantity — identical in kind to how `apply_explosion` converts a blast into an `atmosphere` deposit without storing "explosion temperature." The principle survives intact: **heat crosses air only as radiation (the `heat` buffer); the one thing fire puts into air is pressure, through the atmosphere field, exactly as explosions do.**

**Update the §1 parenthetical** ("future temperature→pressure coupling / thermal-expansion firestorms"): that payoff is now **realised in fire-specific form** by the fire pressure deposit + the existing relief valve — *not* by a general air-temperature field. If a general "hot solid radiates pressure" rule is ever wanted (e.g. a glowing reactor wall with no flame), it is a one-line future deposit `atmosphere[nbr] += k·max(temperature − T0, 0)` reusing the *same* `FieldEdit` + relief-valve machinery — so deferring it costs nothing.

---

## 6. The complete per-tick pipeline (explicit & deterministic)

All thermal work runs in the deterministic sim mutation phase, **after** the read-only ray pass has filled `heat`. Order is load-bearing; rationale inline.

```
PER TICK:

 0. RAY PASS (read-only)         fire/beam/explosion sources cast rays;
                                 march_ray_directional saturating-adds into heat[]   [SHIPPED]
    ── sim mutation phase ──
 1. HEAT → TEMPERATURE (§1.2)    solids only: temperature += heat >> heat_inv_shift   (sat_add)
                                 — inject BEFORE relaxation, so the fresh deposit is on
                                   the field before it spreads (else a hull hit conducts
                                   away before registering locally).
 2. CONDUCTION RELAXATION (§2)   one gather pass, double-buffered: T relaxes toward the
                                 harmonic-mean face-weighted neighbour blend; κ=0 air → no-op.
                                 — spread AFTER inject so "hull hit races down the plate" works.
 3. AMBIENT COOLING (§3)         T -= T >> (exposed ? COOL_SHIFT_VACUUM : COOL_SHIFT)
                                 — AFTER conduction (shed the spread-out reservoir, not the
                                   impact point) and BEFORE consumers (thresholds see net T;
                                   this is what makes fires burn out).
 4. CONSUMERS / unit damage (§4) read the post-cool temperature & the heat[] buffer:
      4a. UNIT HEAT DAMAGE       apply_environmental_damage: max-over-footprint Φ from heat[],
                                 absorb→threshold→env-damage (×zombie mult).   [reads heat[]]
      4b. IGNITION               temp ≥ ignition_temp ∧ O₂ proxy  → start fire
      4c. WALL THERMAL FAILURE   temp > wall_limit → deplete wall_hp (→ destroy_wall)
      4d. SMOKE BURN-OFF         per §5 fire logic
 5. FIRE STEP (§5)               FireSimulation.step(): intensity update; own-tile pressure
                                 deposit into atmosphere (FieldEdit); destroyed tiles →
                                 destroy_wall (patches conductivity/face/vacuum caches via
                                 on_tile_changed).
 6. RECORDER SNAPSHOT            captures post-damage, post-fire state.
 7. CLEAR HEAT (end of tick)     zero heat[] — AFTER every heat consumer (step 1 convert,
                                 step 4a unit damage, render glow sample). The existing
                                 end-of-tick clear simply moves to follow the new consumers.
```

**Determinism of the whole block:** steps 1, 2, 3 are atomic-free gather stencils over frozen input buffers (read old, write new) → bit-identical regardless of traversal order, cross-machine. Step 0's deposit is the shipped order-independent integer saturating-add. Step 4's apply is a serial loop in fixed unit/tile order. No float touches the temperature field at runtime → Level-2 lockstep holds.

---

## 7. New material properties & constants summary

### 7.1 Material table (`[materials]` / `MaterialTable`)

| Column | Values | On the divide? | Notes |
|--------|--------|:--------------:|-------|
| **`thermal_mass`** (NEW) | Hull 64, Steel 64, Glass 32, Door 8, Wood 8, Air 1 | **yes** → powers of two | precomputed at load to `heat_inv_shift = log₂` per-tile cache, parallel to `conductivity` cache |
| `conductivity` (exists) | Hull 50, Steel 45, Glass 1.0, Door 0.3, Wood 0.15, Air 0.0 | no | drives the §2.4 self-shift + §2.5 face table |
| `ignition_temp` (exists) | per material | no | quantized once at load, ΔT above the 20°C reference |
| wall thermal `limit` (exists/per §4c) | per material | no | quantized once at load |
| *(cooling rate — NOT per-material; global, see §3.4)* | — | — | recommended global; future "insulated" = 1-bit flag → larger shift |

### 7.2 New config block — `[physics.thermal]`

| Constant | Value | Role |
|----------|------:|------|
| `TEMP_SCALE` | 65536 | Q16.16, = HEAT_SCALE (shared domain) |
| `SHIFT_AT_REF` | 2 | metal self-rate = ¼ (fastest stable on 4-nbr) |
| `SHIFT_MIN` | 2 | rate floor / stability bound (4×¼ ≤ 1) |
| `KAPPA_REF` | 50.0 | reference conductivity (hull) for the log bucket |
| `NO_FACE` | 63 | sentinel: κ=0 face / grid edge → zero conduction |
| `COOL_SHIFT` | 5 | interior Newtonian cooling: `T -= T >> 5` (~1/32/tick) |
| `COOL_SHIFT_VACUUM` | 3 | space-exposed cool 4× faster: `T -= T >> 3` (~1/8/tick) |
| `o2_vacuum_thresh` | tune | quantized atmosphere below which a neighbour counts as vacuum for exposure |

*(`T_ambient` is implicit = 0: the field is ΔT above a 20°C reference; cooling target is exactly 0.)*
*(`HEAT_INJECT_SHIFT` from the conduction research is **subsumed** by §1's `thermal_mass`/`heat_inv_shift` — the per-material shift *is* the radiation→temperature coupling, so a separate global inject shift is redundant. Drop it; use `thermal_mass`.)*

### 7.3 New config — `[physics.fire]` (temp-adjacent)

| Constant | Suggested | Role |
|----------|----------:|------|
| `fire_pressure_gain` | 0.15 /s per unit intensity | own-tile overpressure deposit (§5.2) — keeps a blaze ~10–30% over ambient, well under a grenade transient |
| `o2_proxy_min` | 0.60 | fire starves below this neighbour-pressure (matches existing `o2_threshold`) |

### 7.4 New config — `[combat]` (unit heat, §4)

| Key | Suggested | Role |
|-----|----------:|------|
| `unit_absorption` (exists) | 0.85 | fraction of incident flux absorbed |
| `unit_reflectivity` (exists) | 0.10 | fraction reflected before absorption |
| `heat_flux_to_temp` | tune | felt-degrees per absorbed heat-unit (`k_flux_to_temp`) |
| `heat_ambient_ref` | band neutral | baseline felt temperature (`T_ambient_ref`) |
| `heat_overtemp_scale` | small +ve | damage ramp above the band (`k_over`) |
| `fire_contact_gain` | tune (deferred) | `k_fire_contact` — only if `Φ_contact` proves needed |
| `solid_contact_gain` | tune (deferred) | `k_solid_contact` — needs temperature field |

*(`environmental_damage_rate`, `zombie.fire_damage_multiplier` already exist, reused unchanged.)*

---

## Open questions for Erik

1. **`thermal_mass` vs a separate `HEAT_INJECT_SHIFT`.** The conduction research proposed a global `HEAT_INJECT_SHIFT = 2` (heat ÷4 → temp); §1 proposes per-material `thermal_mass` (the divide *is* the coupling). I recommend **`thermal_mass` only** (one mechanism, per-material, no redundant global knob). Confirm — or do you want *both* (a global coupling × per-material mass)? That would double-shift and need re-tuning.
2. **8× exaggerated thermal_mass spread** (metal 64 : wood 8) vs the physical ~5×. Good for readability? Or anchor closer to real ρ·c?
3. **Vacuum-exposed faster cooling** (`COOL_SHIFT_VACUUM = 3`, 4× faster) — confirm the direction (breached walls cool fast) and the magnitude. This couples cooling to the vacuum flag the breach path maintains.
4. **`COOL_SHIFT = 5` default** (~32-tick e-fold). Right feel, given it must be slower than a sustained fire's deposit (so a living fire stays above `ignition_temp`) but fast enough to visibly cool within a few seconds after the fire dies? This balance is the burn-out tuning — likely needs the physics demo to lock.
5. **Cooling global vs per-material** — I recommend global modulated only by the vacuum 1-bit (conductivity already encodes material variation via spreading). Agree, or do you want a per-material lever now?
6. **Unit `Φ_contact` term** — ship `Φ_rad`-only first (single buffer read), defer the contact gather until playtesting proves ray-miss ticks matter? Recommended.
7. **Gather floor-bias vs exact per-face flux** in conduction (§2.7). Recommend the gather (≤1 LSB bias, negligible). Confirm we don't need exact energy conservation.
8. **`fire_pressure_gain = 0.15` starting point** — and confirm the temp→pressure decision stays fire-specific (no general field) as recommended in §5.

---

## Build steps (build-FIRST, before fire)

1. **Add `temperature` field** — dense Q16.16 int32 on GameMap, allocated to 0, alongside `heat`. Reuse `HEAT_SCALE`/`heat_saturating_add` from `cpp/src/raycaster.h`.
2. **Add `thermal_mass` column** to `materials.py` + `config.toml`; precompute the per-tile `heat_inv_shift` cache at load, patched in `on_tile_changed` (parallel to `conductivity`).
3. **Heat → temperature convert** (§1.2) — `temperature += heat >> heat_inv_shift` on solids, before relaxation. Move the heat clear to end-of-tick (after all consumers).
4. **Build the face-shift table** (§2.4–2.5) — load-time `self_shift[N]` + `face_shift_table[N][N]` from `conductivity` (log-bucket + harmonic mean); bake the per-tile `face_shift[y][x][dir]` cache, patched in `on_tile_changed`. Add `[physics.thermal]`.
5. **Conduction relaxation pass** (§2.2) — 4-neighbour gather, double-buffered, signed Q16.16, 64-bit accumulator. Verify the discrete-max-principle invariant in a test (no new extremum; air stays bit-exactly 0).
6. **Ambient cooling pass** (§3) — `T -= T >> shift` with the vacuum-exposure 1-bit from the existing atmosphere/vacuum field. Add `COOL_SHIFT`/`COOL_SHIFT_VACUUM`/`o2_vacuum_thresh` to `[physics.fire]`.
7. **Unit heat damage** (§4) — `apply_environmental_damage` serial step (template: `apply_blast_damage`); `Φ_rad`-only first; slot before the recorder snapshot and the heat clear. Add `[combat]` heat keys.
8. **Wire the per-tick order** (§6) and add a determinism test: run the same seeded scenario twice, assert bit-identical `temperature` buffers; assert a fire with cut fuel/O₂ cools below `ignition_temp` and dies (burn-out closes the loop).
9. *(Then fire — §5 temp→pressure deposit lands with the fire mechanic, reusing the `FieldEdit` + relief-valve machinery; temperature is its substrate.)*
