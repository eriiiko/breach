# Smoke-unit rescale: a finer trace currency (design v1, 2026-10-20)

**Issue:** #88. **Branch:** `88-smoke-unit-rescale`, cut off `fire-12` at `c8cf9da`.
**Status:** v1, for critique. Erik delegated #88 to Claude end to end on
2026-10-20 ("you can design it as well as I, and it doesn't need any human
test"), so every decision below is made here, with its reason.
**Out of scope** (Erik's word): the soot AMOUNT (`soot_yield`, the studio's
2–4 m visibility). Erik revisits it after #88.
**Depends on:** `docs/smoke_transport_design_2026-10-04.md` (v2 transport,
D1–D6) and `cpp/src/trace_decay.h`.

## 0. Summary

- The five trace planes (steam, smoke, poison, teargas, fuel_gas) get
  **`TRACE_FINE_BITS = 8`** more fractional bits: one unit of trace density
  becomes `2^24` counts instead of `2^16`. The int32 storage, the transport
  law, every config value and every reader's output in real units stay as they
  are. Only the integer currency of the trace planes changes. The bulk planes
  (O2, inert_N2) keep Q16.16.
- The factor is the measured compromise between two needs (§3). It gives
  128 units of per-cell headroom, which is 440 times the largest real per-cell
  value and 3.3 times the worst legal abuse measured. Decay then follows the
  configured exponential down to the edge of visibility.
- Three things the rescale exposes are fixed in their own patches:
  1. The pump/vent extraction split mixes trace counts with bulk counts (P0).
  2. Every trace int32 narrow either raises or silently wraps once a cell can
     reach `INT32_MAX`; a counted rail replaces that (P2).
  3. The ceil decay rounding biases the faint end; it is amended to
     nearest-with-a-floor-of-1 (P5).
- One constant, one owner per language, and twins held equal by a test. The
  ambiguous `gas_fixed.quantize` / `dequantize` / `FP_ONE` / `SMOKE_MAX_Q` names
  are **deleted** and replaced by `bulk_*` and `trace_*` names, so a Python
  site that was missed fails loudly rather than being silently 256 times off.
- Proof that the change is ONLY the rescale: the whole code is written in
  terms of `TRACE_FINE_BITS`, and at `TRACE_FINE_BITS = 0` the tree reproduces
  today's `GOLDEN_AGGREGATE` (`f6ba810e…`) bit for bit, on CPU and CUDA (P1).
  Flipping the constant to 8 (P3) is then the only change behind the
  re-baseline.

## 1. The problem, restated with numbers

Visible smoke lives in the bottom ~11 bits of the int32 plane. One Q16 count
of smoke is τ = 134.29 / 65536 ≈ 0.0020 per tile (green, `[gases.smoke]
light_absorb`), so "faintly visible" (τ ≈ 0.01) to "black" (τ ≈ 4) is 5 to
2000 counts. At that resolution, three laws break:

| law | integer form | threshold at Q16 | effect at Q16 |
|---|---|---|---|
| decay | `lost = ceil(v·frac_q / 2^16)`, `frac_q = 22` (0.008/s at 24 tps) | exponential needs `v ≫ 2979` counts, i.e. τ ≫ 6 | every visible tile loses 1 count a tick: a straight line to zero. Opaque, then gone at once (Erik's report) |
| diffusion | `|F| = (c·|ΔS|) >> 16`, `c = dd_q = quantize(0.10/24) = 273` | `|ΔS| ≥ 240` counts, i.e. Δτ ≥ 0.49 per tile | still-air spreading is DEAD across almost the whole visible range |
| poison near its threshold | the same decay | `poison_min_density = 0.05` = 3277 counts, under poison's own exponential threshold (5958) | poison at the dose edge decays linearly, twice its configured rate |

Erik's ruling (2026-10-05): the configured decay (`[gases.smoke] decay`, a
125 s e-fold) is the rate the game must actually show.

## 2. Inventory: every writer and reader of the trace planes

Collected by grep over the worktree, 2026-10-20. "Converts" says what the
site must do after the rescale. **INLINE** marks a site that today applies
`>> 16`, `/ 65536` or `FP_ONE` to a trace value; after #88 every such site goes
through a kit function (§4).

### 2.1 Cross-cutting facts

- `GameMap.gas` is one `(N_GASES, h, w)` int32 array (`gamemap.py:360`):
  trace ids 0..4, bulk ids 5..6. `gmap.smoke` is a view (`:381`). After #88
  the array holds **two currencies**. That is the main hazard, so every site
  that treats all seven planes as one unit is listed in §2.2 and removed
  first.
- `gas_fixed.py` serves BOTH the trace planes and the bulk planes, and other
  systems too:
  - O2/N2 seeding (`gamemap.py:873-885`, `:1384-1395`, `:3040-3045`)
  - `field_edit._combine_atmosphere_to_N` (`:463-473`)
  - the T_AMB fold (`gamemap.py:925`, `physics_runner.py:1835`)
  - the pump ΔN (`pump_system.py:142-147`)
  - filter fractions (`vent_system.py:120-128`)
  - the sky λ (`physics_runner.py:1287`)
  - `ambient.py`, `temperature_scale.py:33`

  Its constant cannot simply change; it must split (§4).
- numpy is 2.4.6 on this machine, so `int32_array.sum()` accumulates in int64.
  An explicit `dtype=np.int32` accumulator anywhere is a bug after #88; P3
  greps for it.

### 2.2 Sites that MIX trace and bulk counts (must be removed before the flip)

| site | what it does | fix |
|---|---|---|
| `gamemap.py:3081-3115` `extract_gas_n_vec`, and its callers `extract_gas_n` (`pump_system.py:113`) and `vent_system.py:390-391` | splits a bulk budget `want` over the holdings of **all seven** planes (`gas_proportional_split`), so trace counts are spent as if they were air | **P0**: the split runs over the bulk planes only; each trace plane leaves **priced by air**, `take_t = floordiv(bulk_taken · S_t, N_bulk)`. That is the ratio law stage 3b and the vent supply side (`vent_system.py:476`) already use, so it is scale-invariant. The scalar `extract_gas_n` returns the bulk total |
| `gamemap.py:3151` `inject_gas_n_vec` `dn_total = sum(add_vec)` | a `> 0` gate only | harmless, but sums the bulk slices only (P0) |
| `combat.py:1131-1163` `fire_beam` | loops over all `N_GASES`, `(absorb[g]·S) >> 16` | loops over the trace ids; the shift goes through the kit (P1) |
| radiation sweep, `radiation_sweep.cpp:507-522, 595-606, 664-677` (+ CUDA `:175, 263-265, 311-318, 956-975`) | the density sums run over every gas with a non-zero coefficient | **the gases door refuses a non-zero `heat_absorb`, `absorption`-derived light/beam coefficient or `scatter_albedo` on a conservative (bulk) row** (P1). The sums therefore only ever see trace planes and are in the trace currency. Every shipped bulk row is 0 today, so nothing ships differently (ACCEPTED GAP G3) |
| `hover_readout.py:161-164` | dequantizes STEAM..FUEL_GAS **and O2** through one `_GAS_FP_ONE_F` | a per-plane currency: trace ids through `dequantize_trace`, O2 through `dequantize_bulk` |
| `hover_readout.py:241-245` | all seven planes go to `gas_extinction_q16` | unchanged, because the door makes the bulk coefficients 0 |
| `src/debug_keys.py:155-168` `debug_spawn_gas` | writes `SMOKE_MAX_Q` into any gas id, bulk included | writes `TRACE_ONE` on a trace id and `BULK_ONE` on a bulk id |
| `gamemap.py:2662-2670` (`_seal_blockers`), `:3010-3017` (`_inject_gas_n_core`) | `limit = 2^31` OverflowError per slice | the trace slices take the rail (§5.3); the bulk slices keep the error |

### 2.3 Writers

| writer | where (CPU / CUDA) | today | after #88 |
|---|---|---|---|
| FieldEdit trace ADD/REMOVE/MAX, the door for `emit_gas`, `add_explosion_smoke`, `blast_smoke_peak`, spray cones, `physics.py` smoke clear | `field_edit.py:499-510` `_combine_gas` (**INLINE** `/ FP_ONE_F`, `quantize_scalar`) | dequantize the tile, add in float, re-quantize | the same combine (dequantize the tile through `dequantize_trace`, combine in float64 under the mode's semantics, re-quantize through `quantize_trace_scalar`), then the rail (§5.3). This is exact in float64 for any int32 tile at 2^24 (|v| < 2^31 ≪ 2^53), so k = 0 is the old code path verbatim. The real-unit amounts are unchanged; deposits round 256 times finer |
| combustion soot mint | `combustion.cpp:1003-1005`, `cuda_combustion.cu:518-520` | `soot = narrow_round(burn·soot_yield_q)`, then `SOOT += soot`. `soot` is in **bulk** counts and is also subtracted from N2 and `e_shed` | `SOOT += trace_from_bulk(soot)` (`<< TRACE_FINE_BITS`), then the rail. The Dalton split and `e_shed` keep the bulk-count `soot`, so the bulk and energy books are bit-identical and the minted smoke is exactly the old × 2^8 |
| W5 steam puff | `physics_engine.cpp:1044-1045` (the resident path runs this CPU tail) | `puff_q = quantize(steam_yield·boiled_m)` | `quantize_trace(...)`, then the rail |
| W3 water evacuation | `physics_engine.cpp:1073-1140` | `evac = mul_q16(plane, frac_q)`, shared to neighbours | scale-invariant (a fraction of a trace count). The receiver add takes the rail |
| stage 3b advection | `bulk_transport.cpp:457-548`; `cuda_bulk_transport.cu:403-476` `trace_apply` (chained and resident) | gather; int32 narrow with `assert` (CPU) / no check (CUDA) | `price_face` is a ratio, so unchanged. The narrow takes the rail |
| stage 3c wipe | `bulk_transport.cpp:619-641`; `cuda_bulk_transport.cu:483-498` | sub-N_EPS cells zeroed, booked | unchanged |
| trace tail: sink, diffusion, decay | `bulk_transport.cpp:681-809`; resident `cuda_bulk_transport.cu:834-962` | sink zeroes; Jacobi `(c·|ΔS|) >> 16`; ceil decay | unchanged in form. `>> 16` there is a rate shift on a count, so it is scale-invariant. Diffusion is a convex combination (`Σ c ≤ ONE`), so it cannot exceed the plane's max and needs no rail. Decay rounding amended in P5 |
| vents (plenum extract / supply inject) | `vent_system.py:390-520` | priced by air on supply; mixed split on extract | P0 split; supply inject through `inject_gas_n_vec` takes the rail |
| `seal_tiles` / `unseal_tiles` redistribution | `gamemap.py:2769-2780`, `:2896-2917` | divmod per slice, scale-invariant, overflow → error | the trace slices take the rail |
| `destroy_wall` | `gamemap.py:2298+` | bulk only; the trace on the tile is sunk by the tail | unchanged |
| debug spawn | `src/debug_keys.py:155-168` | `= SMOKE_MAX_Q` | `= TRACE_ONE` (§2.2) |
| level load / hot reload | `gamemap.py:884-885`, `:1574-1575`, `:1793-1796` | no level authors trace gas; reload copies the array | unchanged. No shipped level carries trace values |
| tools that seed trace | `tools/lighting_demo.py:916-930` (`SMOKE_MAX_Q·amount`), `tools/eos_p5_bake.py:68-71, 227, 285` (**INLINE** `q16()`), `tools/bench_radiation_sweep.py:127-133` (raw Q16 fixture) | Q16 | through `quantize_trace` |

### 2.4 Readers

| reader | where | today | after #88 |
|---|---|---|---|
| sweep heat term `a_gas` | `radiation_sweep.h:248-252` `gas_extinction_finish` (**INLINE** `sum >> FP_SHIFT`); call sites CPU `:604`, CUDA `:265`, binding `bindings.cpp:2650-2680` | `min(ONE, Σ hq·S >> 16)` | `min(ONE, trace_density_shr(Σ hq·S))` = `>> (16 + TRACE_FINE_BITS)`. With `S_new = S_old·2^8` the result is **bit-identical**: `⌊c·S·2^k / 2^(16+k)⌋ = ⌊c·S / 2^16⌋` |
| sweep light term + glow coefficient | `radiation_sweep.h:274-277` `gas_extinction_exp_finish` (**INLINE**); CPU `:664-677`, CUDA `:311-318` | the same density sum `>> 16` | the same kit shift. The glow `mul128_shr(light_q, g_c, 16)` reads the fraction `g_c`, so it is unchanged |
| integer reference | `sweep_ref_q.py:1047-1121` `gas_density_sum`, `gas_extinction_q`, `gas_extinction_exp_q` (**INLINE** `>> 16`); `:2019` glow | spec | changed FIRST (§7) |
| laser | `combat.py:1157` (**INLINE** `>> FP_SHIFT`) | per-gas floored term | `trace_density_shr(absorb·S)` per term. Bit-identical on `S_old·2^k` |
| poison dose | `exchange.py:670, 681` (**INLINE** `/ 65536.0`) | `quantize_scalar(threshold)`; `dps·S/65536·dt` | `quantize_trace(threshold)`; `dps·dequantize_trace(S)·dt`. The float is identical on `S_old·2^k` (division by a power of two is exact) |
| teargas BLINDED | `exchange.py:625` | `quantize_scalar(0.15)` | `quantize_trace(0.15)` |
| smoke sensor | `sensor_accessor.py:130-132` → `sensor_system.py:86-92` → decider `KIND_Q16` thresholds (`entities/nodes.py:94`) | raw Q16 counts published | the accessor converts once, `trace_to_q16(S) = S >> TRACE_FINE_BITS` (floor). The signal contract stays Q16 density, so authored decider thresholds keep their meaning. No shipped level wires a smoke sensor |
| recorder | `recorder.py:190-194` (**INLINE** `/ 65536.0`), float32 ring of `smoke` | real units | `dequantize_trace_f32`. The recorded float is still real units, so the frozen `.npz` contract and old recordings are unchanged |
| renderer | `game_renderer.py:479, 486, 510, 578-579` (`dequantize_f32` of trace planes) → `gas_medium`, `gas_relief`, `gas_detail`, speckle, overlays, `lighting`, `lit3d` | float units | `dequantize_trace_f32`. Everything downstream already works in floats |
| `gas_medium.py:101-104` | `light_absorb_q16 / 65536` | a COEFFICIENT, not a density | unchanged: coefficients stay Q16 per real unit (§4) |
| hover readout | `hover_readout.py:161-164, 241-245` | §2.2 | §2.2 |
| books | `bindings.cpp:2966-2993` `boundary_flux()` (trace slots in trace counts beside bulk slots), `trace_wipe_sum`, `trace_sink_sum`, `trace_decay_sum` | counts | **counts in each plane's own currency**; documented on the binding |
| vent entity digest | `entities/vents.py:117-120` `trace_{i}` / `sink_{i}` rows | trace counts, Python ints | values move ×2^8 (ENTITY_SECT bytes move; no format change). P3 checks whether `serialize.py` persists the plenum; if it does, the serialized form records the currency |
| tools / benches | **INLINE**: `tools/bench_clamp_shave.py:460`, `tools/storm_ledger.py:132-145`, `tools/eos_p5_bake.py:237, 292, 399, 421`; via `gas_fixed`: `tools/wind_drift_probe.py:78`; coefficient only: `tools/smoke_tau_histogram.py:63` | Q16 | converted through the kit; the coefficient-only one is unchanged |
| not readers (checked) | `fire_simulation.cpp:95-98` / `cuda_fire.cu:356` (smoke `(void)`), `temperature_solver` (reads the sweep's `rad_net`), `sky_exchange.cpp`, `water_solver`, `light_field.py`, `status.py`, `vision.py`, `ai_zombie.py` | — | — |

### 2.5 Scale-free sites (confirmed, no change)

- `price_face(dq, S, N)`: a ratio, `dq·S < 2^30·2^31 = 2^61`.
- `trace_diffusion_dd_q` and its stability door; `trace_decay::frac_q`: rates.
- Vent filter scrub `(r·filter_eff_q) >> 16`: a fraction of a count.
- `vent_system.py:476`: priced by air.
- `seal_tiles` divmod; W3 `mul_q16(plane, frac_q)`.

### 2.6 Config keys denominated in trace density

These are all REAL units, unchanged by #88, converted at the doors above:
- every `gas_amount` (`config.toml:2808, 2825, 2907, 2976, 3012, 3029, 3115, 3120, 3125`)
- `[payloads.*] soot_yield` (g/g), through `smoke_units_of_soot`
- `[physics.combustion] soot_yield`
- `[physics.water] steam_yield`
- `teargas_blind_density`, `poison_min_density`, `poison_dps`
- the `[gases.*]` `absorption` / `heat_absorb` / `light_absorb` / `scatter_albedo`
  columns, which are per real unit

So `soot_g_per_smoke_unit`, `smoke_units_of_soot`, `test_gas_heat_absorb.py`
and `test_gas_light_absorb.py` do **not** change. The issue's list of
"everything denominated in smoke units" was written for a rescale of the UNIT
(×100/×1000 meaning). A rescale of the CURRENCY by a power of two leaves the
unit's meaning alone, which is why the factor is a power of two.

## 3. The factor, chosen by measurement

### 3.1 What real scenarios produce

Measured 2026-10-20 on this worktree's CPU build (`fire-12` `c8cf9da`), with
the script `measure_trace.py` in the session scratchpad. Each tick the script
records every trace plane's per-cell max and grid total, plus the bulk-N max.
Fires are lit with `debug_ignite`'s own law on every flammable tile.

| scenario | sim time | per-cell max (units) | grid total max (units) | bulk N max (× ambient) |
|---|---|---|---|---|
| smoke_light_studio, 3 charges | 40 s | 0.256 | 8.66 | 10.7 |
| explosion_studio, 4 charges every 5 s | 180 s | 0.291 | 9.04 | 18.0 |
| playground, every flammable tile lit | 400 s | 0.025 | 3.61 | 18.0 |
| test_level (256×128, the largest map), every flammable tile lit | 400 s (peaked by 30 s) | 0.034 | 7.93 | 3.0 |
| fire_studio, all lit | 300 s | 0.025 | 0.34 | 1.2 |
| fire_tuning, all lit | 300 s | 0.019 | 0.54 | 1.3 |
| smoke_screen grenade core (by its row) | — | 1.5 | — | — |
| **abuse:** 10 smoke_screen grenades on one tile, 1 s apart, sealed 18×26 room | 60 s | **10.95** | 242 | 1.0 |
| **abuse:** a Miasma Vent's 0.35/tick held on ONE tile of a sealed 18×26 room | 1500 s (equilibrium) | **38.85** | 2062 | 1.0 |

Reading the table:
- Real play stays below 0.3 units per cell, and grid totals stay below 10
  units.
- Fire makes very little smoke at today's yield. If Erik raises it, the
  combustion fraction is capped at 1.0, i.e. at most twice today's.
- The two abuse cases are legal, and RL agents will find them.
- The miasma equilibrium is `rate / decay` = `0.35·24 / 0.004` ≈ 2100 units
  in the room. In a 3×3 closet that would be about 230 units per cell. Bulk N
  reaches 18 times ambient in a blast, and stage 3b's ratio-preserving
  transport compresses the trace with it. So **no factor above about 4 bits
  is safe from int32 overflow under abuse. A counted rail is required at any
  useful factor (§5.3).** The factor only decides how far beyond real play
  the rail sits.

### 3.2 What the visible range needs

Here is the decay law on one smoke cell (`decay_law.py` in the scratchpad):
an integer cell started at τ = 4 per tile, against the configured e-fold. The
table gives the times to cross each τ band, in seconds; "10 % band" is the τ
above which every value's per-tick loss is within 10 % of the exact
exponential.

| k | rounding | 4→1 | 1→0.1 | 0.1→0.03 | 0.03→0.01 | 0.01→0 | 10 % band from τ |
|---|---|---|---|---|---|---|---|
| — | **configured exponential** | 173 | 288 | 150 | 137 | ∞ | — |
| 0 | ceil (today) | 61 | 18 | 1 | 0 | 0 | 55 |
| 8 | ceil | 171 | 273 | 123 | 88 | 52 | 0.22 |
| 8 | **nearest, floor 1 (P5)** | 172 | 286 | 155 | 104 | 52 | 0.11 |
| 9 | nearest, floor 1 | 172 | 286 | 149 | 145 | 104 | 0.054 |
| 10 | nearest, floor 1 | 172 | 286 | 150 | 137 | 197 | 0.027 |
| 12 | nearest, floor 1 | 172 | 286 | 149 | 136 | 370 | 0.0068 |

Today's law takes 80 s from black to nothing, against about 750 s configured
before the last trace. That is the reported "opaque, then gone".

### 3.3 Headroom per factor

The per-cell int32 ceiling is `2^31 / 2^(16+k)` units:

| k | ceiling (units) | × real max (0.29) | × grenade spam (11) | × miasma equilibrium (39) |
|---|---|---|---|---|
| 8 | 128 | 440 | 11.7 | 3.3 |
| 9 | 64 | 220 | 5.8 | 1.6 |
| 10 | 32 | 110 | 2.9 | **0.8: the rail binds in measured legal play** |
| 12 | 8 | 28 | **0.7** | **0.2** |

### 3.4 Decision: `TRACE_FINE_BITS = 8`

- **A power of two**, so the unit's meaning never changes, every reader's
  result on `S_old·2^k` is bit-identical (§2.4), and `k = 0` IS today's
  engine (§6).
- **k = 8** keeps the rail out of every measured legal case, including 25
  minutes of a Miasma Vent held on one tile, with a 3.3× margin. k = 9 halves
  that margin to 1.6×. k ≥ 10 lets the rail bind in measured legal play.
- With P5's rounding, **k = 8 follows the configured decay within 3 % from
  black down to τ = 0.03 per tile** (3 % opacity per tile), and within 25 % to
  τ = 0.01 (1 %). The last 1 % haze then leaves in 52 s instead of never.
  Nothing a player can see goes opaque-then-gone.
- k = 9 would buy the 0.01–0.03 band (1–3 % opacity per tile, the edge of
  visibility) at the cost of half the abuse headroom. The headroom is worth
  more in an RL environment.
- **The arithmetic of the issue, redone at k = 8:**
  - A just-visible tile (τ = 0.01) holds 1250 counts and loses 0.42 counts a
    tick exactly. Nearest rounding with a floor of 1 removes 1: the edge of
    the linear regime.
  - At τ = 0.03 it holds 3750 counts, the loss is 1.26, and the rounding
    removes 1 (−21 % instantaneous, averaging out over the band).
  - At τ = 0.1 it holds 12 500 counts, the loss is 4.2, and the rounding
    removes 4.
  - Diffusion acts from |Δτ| ≥ 0.0019 per tile, against 0.49 today.
  - Poison at its 0.05 dose threshold holds 838 861 counts. Its loss is 141
    counts a tick: fully exponential.

## 4. Where the scale lives

**One constant, one owner per language, twins held equal by a test**, in the
pattern of #78's `E_FINE_BITS`.

**C++** (`cpp/src/fixed_point.h`, namespace `fixedpoint`, FP_HD, the one kit
both backends compile):

```
constexpr int TRACE_FINE_BITS = 8;                      // #88: the trace planes' currency
constexpr int TRACE_SHIFT     = FP_SHIFT + TRACE_FINE_BITS;   // 24
constexpr int32_t TRACE_CELL_MAX = INT32_MAX;           // the counted rail (§5.3)
FP_HD int64_t trace_from_bulk(int64_t q);               // q << TRACE_FINE_BITS (a bulk-count mass entering a trace plane: the soot mint)
FP_HD int64_t trace_density_shr(int64_t sum);           // sum >> TRACE_SHIFT  (Σ coef_q16 · S  ->  Q16 fraction; sweep, laser, readout)
FP_HD int32_t trace_add_rail(int32_t s, int64_t add, int64_t* clip);  // saturating add, excess booked
quantize_trace(double)                                  // round-half-away at 2^TRACE_SHIFT (the W5 puff); host only
```

Every C++ site in §2 calls these. No `<< 8`, `>> 24` or `>> FP_SHIFT` on a
trace value is written anywhere else. `/fp:strict` already covers
`fixed_point.h`'s users, and the kit adds no float in a sim TU except
`quantize_trace`, whose door-2 idiom matches `quantize`'s. Bound by the
binding as `breach_physics.TRACE_FINE_BITS`.

**Python** (`src/simulation/gas_fixed.py`, the Q16 boundary module for gas):
- `TRACE_FINE_BITS = 8`, `TRACE_SHIFT`, `TRACE_ONE = 1 << 24`,
  `TRACE_CELL_MAX`.
- `quantize_trace`, `quantize_trace_scalar`, `dequantize_trace`,
  `dequantize_trace_f32`, `trace_density_shr`, `trace_to_q16`, `trace_add_rail`.
- The bulk side: `BULK_ONE`, `quantize_bulk`, `quantize_bulk_scalar`,
  `dequantize_bulk`, `dequantize_bulk_f32`.
- **Deleted:** `FP_ONE`, `FP_ONE_F`, `SMOKE_MAX_Q`, `quantize`,
  `quantize_scalar`, `dequantize`, `dequantize_f32`. Every caller (§2.1's
  bulk users included) is moved to the name of its own currency, and a missed
  one raises `AttributeError`.

**Reference** (`sweep_ref_q.py`): `TRACE_FINE_BITS` and
`trace_density_shr`, guarded equal to the engine's (§7).

**Digest spec** (`tests/field_digest_spec.toml`): `trace_fine_bits = 8`
(§6).

**Coefficients stay Q16 per REAL unit**: `heat_absorb_q16`,
`light_absorb_q16`, `light_glow_q16`, `beam_absorb_q16`. The density laws
absorb the extra bits in their one shift (`trace_density_shr`), not in the
coefficient. That way:
1. the coefficients keep their full precision (a Q16 `light_glow` of
   0.22 / 256 would be 56 counts, a 1.8 % rounding);
2. the door's `HEAT_ABSORB_MAX` headroom argument is untouched (each term
   `< 2^28 · 2^31 = 2^59`, 16 terms `< 2^63`; S is still int32);
3. `gas_medium.py`'s `/65536` on a coefficient stays correct.

**Guard test** (P1): `breach_physics.TRACE_FINE_BITS ==
gas_fixed.TRACE_FINE_BITS == sweep_ref_q.TRACE_FINE_BITS == spec
trace_fine_bits`. It also lints that no `src/`, `renderer/` or `tools/`
file references the deleted names, and that the four §2.4 INLINE C++ sites
call the kit.

## 5. Conservation and books

### 5.1 The identity, extended by one channel

Today (v2 §3), per tick and plane, exactly in int64:

    Σ S(after) − Σ S(before) == deposits − vent − wipe − sink − decay

After #88, in each plane's own currency (trace counts for the trace planes):

    Σ S(after) − Σ S(before) == deposits − vent − wipe − sink − decay − clip

`clip` is the new rail channel (§5.3), and it is zero in every measured
legal scenario.

### 5.2 Why every existing book stays exact

| channel | law | exact after #88 because |
|---|---|---|
| advection (3b) | debit and credit of a face are the SAME `price_face` int64 | unchanged law; S is still int32, products `< 2^61` |
| vent | the donor books what it prices or diffuses onto vacuum/ring | the same int64, now in trace counts |
| wipe (3c) | the whole S of a sub-N_EPS cell | unchanged |
| sink | the whole S of a stranded cell | unchanged |
| diffusion | `F(i,j) = −F(j,i)`, magnitude-truncated, `c·|ΔS| < 2^47` | unchanged |
| decay | `Σ lost`, `v·frac < 2^47` | unchanged in form (P5 changes the rounding, not the booking) |
| books' range | reset every tick (`eos_solver.cpp:351-357`); worst case a full map at `INT32_MAX` | `2^15` cells × `2^31` = `2^46` per tick, far below `2^63` |
| bulk/energy books | the soot mint's Dalton split and `e_shed` keep the bulk-count `soot` | bit-identical (§2.3) |

The tail's debug-only assert `Σ_map S < 2^31` (`bulk_transport.cpp:802-807`)
is **deleted**, not re-derived. At k = 8 it is false in measured legal play
(the miasma room holds 2062 units, against 128 at 2^31). It stopped being the
bound when D1 made deposits unbounded, and the rail replaces it as the
argument for the narrow.

### 5.3 The rail: a counted saturation at `TRACE_CELL_MAX = INT32_MAX`

**Why it exists.**
- Today's narrows `assert` in Debug and wrap in Release
  (`cuda_bulk_transport.cu:473` does not even assert). The Python guards raise
  `OverflowError` mid-tick (`seal_tiles` would refuse to close a door on a
  smoky tile).
- At k = 8 a cell can legally pass `2^31`: abuse deposits, compressed
  18 times by a blast.
- A wrapped trace cell goes negative: the sweep reads it as clear and the
  poison dose as nothing. For an RL environment that is a found exploit.

**What it is.** One kit function per language, `trace_add_rail(s, add,
&clip)`:
- `s' = min(s + add, TRACE_CELL_MAX)` computed in int64;
- the excess goes into a per-gas int64 `clip` book;
- it never produces a negative;
- it is pure integer and FP_HD, so it is identical on CPU and CUDA.

**Where it sits.** Every site that can add to a trace cell:

| site | rail |
|---|---|
| FieldEdit trace ADD | `field_edit._combine_gas`, booked in `GameMap.trace_clip_books[g]` (Python ints) |
| stage 3b narrow | `bulk_transport.cpp:540-543`, `cuda_bulk_transport.cu:473`, booked in `EOSSolver.trace_clip_sum` (CUDA: the unsigned-long-long `atomicAdd` idiom, beside the existing trace counters) |
| soot mint | CPU + CUDA → `trace_clip_sum`, through a pointer PhysicsEngine wires, as it does `boundary_flux_` |
| W5 puff and the W3 receiver add | → `trace_clip_sum` |
| `seal_tiles` / `unseal_tiles` / `_inject_gas_n_core` (vents' supply inject) on trace slices | → `GameMap.trace_clip_books` |

Diffusion is convex and decay only removes, so neither needs the rail; their
asserts stay.

**Its relation to the rules.**
- It is NOT a ceiling in the sense of D1 or the Trace transport rule ("no
  trace plane is clamped anywhere outside a reader"). It is the
  representability limit of the int32 plane, 128 units, 440 times the real
  maximum, and it books every count it drops.
- The Trace transport rule is amended to name it (§11).
- Erik's D1 intent, "smoke adding is additive", holds for every value the game
  can show.

## 6. Determinism and goldens

**What moves at k = 8.**
- **The field digest's `gas` bytes**, in every scenario with trace. It
  carries the trace planes (`field_digest.py:102`).
- **`GOLDEN_AGGREGATE`.** Its canonical scenario seeds smoke 0.6
  (`field_ab_harness.py:136`), and the 0.6 now quantizes at 2^24:
  `10066330 ≠ 39322·256`. Through `a_gas`, `temperature` and `gas_energy`
  can move with it. The re-baseline note names the fields that moved and
  their first tick, in the standing format.
- Every CUDA check that reads `GOLDEN_AGGREGATE`.
- The vent ENTITY_SECT rows.
- **Nothing else**: no float field and no unit state move unless the smoke
  reaches them.

**Spec version → 7.** The spec's procedure names dtype, shape, membership and
order. None of those move, but a v6 and a v7 digest of the same world are not
comparable. That is exactly what the version exists to refuse.
- So v7 adds `trace_fine_bits = 8` to the spec.
- `field_digest.py` asserts it equals `gas_fixed.TRACE_FINE_BITS` at import.
  Any later change of the currency is then forced through a version bump.
- The serialization is unchanged.

**Proof that the change is ONLY the rescale (P1 → P3):**
1. **k = 0 is today's engine, bit for bit.** P1 lands every conversion,
   the kit, the deleted names, the door restriction and the reference change
   with `TRACE_FINE_BITS = 0` in all twins. That build must reproduce
   `GOLDEN_AGGREGATE` = `f6ba810e…` exactly, and the whole CPU suite and
   every CUDA check must be green. This holds unless P0 moved the golden, in
   which case P1 reproduces P0's value.
   - At k = 0 every kit function is the old expression: `<< 0`, `>> 16`, and
     quantize at 2^16.
   - FieldEdit keeps its float64 round-trip, only renamed (§2.3).
   - P2's rail does not bind.
   - So any difference at k = 0 is a refactor bug.
2. **The flip is one constant.** P3 changes `TRACE_FINE_BITS` from 0 to 8 in
   the four twins, plus the test fixtures that wrote raw Q16 trace. The
   re-baseline's VERIFIED-SOLE-CAUSE paragraph is then the standing one: the
   same tree at 0 reproduces the previous value.
3. **Reader invariance**, a standing test (V3, §10): a plane seeded with
   `S_old << k` gives bit-identical outputs from every integer reader (`a_gas`,
   light `a`, glow coefficient, laser transmittance, BLINDED gate, sensor
   signal) and identical floats from the float readers (poison dose, recorder,
   render dequantize) to the pre-#88 law evaluated on `S_old`. This catches
   any reader that missed its conversion, which k = 0 cannot see.

**Where new is ALLOWED to differ from old × 2^8, and why:**

| difference | why it is allowed |
|---|---|
| FieldEdit, W5 and the reference's seeds quantize at 2^24 | finer rounding of the same real amount, `|new − old·2^8| ≤ 2^7` counts per edit |
| `price_face`, diffusion and W3 floors | each floors at the fine currency, so they lose up to 2^8 times less |
| decay | the purpose of #88 (the old law is linear in the whole visible range) |
| diffusion acting where it was dead (|ΔS| < 240 old counts) | the same law, now resolved; a visible behaviour change: smoke spreads in still air at visible densities (D-B) |
| everything downstream of those through `a_gas` (temperature, gas_energy), the light, and poison/teargas timing | follows from the rows above |

Bulk N, wind and `gas_energy` stay bit-identical with or without trace
(design v2's V5), unchanged, because the trace still never touches the air.

## 7. The integer reference

The sweep's gas term reads the trace planes, so per the reference's rule
`sweep_ref_q.py` changes FIRST, in P1, before the C++:
- `TRACE_FINE_BITS` and `trace_density_shr` are added.
- `gas_extinction_q` and `gas_extinction_exp_q` shift by `16 +
  TRACE_FINE_BITS`.
- The glow coefficient follows: it is `gas_extinction_q` over `light_glow_q`.
- `gas_density_sum`'s term bound is unchanged, because S is still int32.
- **The gates' scenes must stay non-vacuous.** G13, G14, G15, G18 and G19
  draw random smoke "from none to thick" in Q16 counts. At k = 8, Q16-sized
  densities would make `a_gas` 256 times smaller, and the gates would quietly
  stop exercising absorbing cells. So `_gas_planes` draws in the trace
  currency (`trace_from_bulk` of today's draw), and each gate keeps its
  printed count of absorbing cells, which must stay above zero.
- Gates re-run in P1 at k = 0 (identical output) and in P3 at k = 8 (all PASS).
- `tests/test_radiation_sweep_reference.py` (gate 0, C++ == reference bit for
  bit) and `tests/test_temperature_gas_radiation.py` follow.
- A guard asserts the reference's `TRACE_FINE_BITS` equals the engine's (§4).

## 8. CUDA

- **One kernel serves both chained and resident** for stage 3b/3c
  (`bulk_flux_energy_transport_device`). The resident tail
  (`trace_tail_sink` / `_diffuse` / `_decay`) needs no unit change, only the
  P5 rounding through `trace_decay.h`.
- The soot mint (`cuda_combustion.cu:518-520`) and the sweep's gas terms
  (`cuda_radiation_sweep.cu:263-265, 311-318`) call the same FP_HD kit
  functions as the CPU.
- The rail's book is one more per-gas slot in the existing trace counter
  block (`cuda_eos_resident.cu:930-943`, `cuda_eos_step.cu:144-155`), D2H'd
  with the others.
- **Gates at tolerance 0** through the CUDA harness (`run_cuda_script`, never
  importing the CUDA .pyd into pytest): `tests/cuda_trace_smoke_check.py`
  (chained and resident, every plane and counter including clip),
  `cuda_combustion_check.py`, `cuda_radiation_sweep_check.py` (light
  included), `cuda_s8a_check.py`, and the thermal-mass-eos resident leg.
  Every `GOLDEN_AGGREGATE` CUDA consumer is re-greened at P1 (unchanged value)
  and at P3 (new value).
- **A CUDA rail test:** a cell seeded at `INT32_MAX − 10` receiving inflow
  gives GPU == CPU on the plane and on `clip`.

## 9. Look-side consequences (render only: listed, not tuned)

The renderer reads dequantized floats, so its inputs keep their units. What
changes is the field it is shown: smoke now thins gradually and spreads in
still air. The smoke-tuning skill's §6 list stands. Retune once #88 is in:
- `cov_tau_*`, `cov_soft_*`, `haze_*`, `height_tau` and `top_light_*`
  (possibly to 0), all set on a field that went opaque and then vanished.
- Recheck the boil range after #12's fire tuning.
- Two additions for that list:
  - **the speckle and `gas_relief` clips**, which saw only coarse 2^-16 steps
    at thin densities;
  - **`effect_gas_floor`**, the poison/teargas legibility floor, because
    poison now lingers at its configured rate near the dose threshold.

The doc for the list stays `docs/graphics_direction_2026-10-18.md` §11. P6
adds a dated line to the skill.

## 10. Verification plan (properties)

Each item names the property and the change that must break it. Nothing pins
a count, a set of gases, or a snapshot value.

- **V1, twins agree.** C++, Python, the reference and the digest spec name the
  same `TRACE_FINE_BITS`, and the deleted `gas_fixed` names are referenced
  nowhere. *Breaks if* one twin moves alone or an ambiguous name returns.
- **V2, k = 0 is today (P1's report, not a standing test).** The P1 tree at
  `TRACE_FINE_BITS = 0` reproduces the previous `GOLDEN_AGGREGATE` on CPU and
  CUDA. *Breaks if* any conversion is not the old expression at k = 0.
- **V3, reader invariance.** Seed every trace plane with `S_old << k` for a
  sweep of `S_old`. This covers: densities at, below and above each threshold;
  the ONE cap of `a_gas`; and a negative cell, which the sweep treats as 0.
  - Integer readers (`a_gas`, light `a_c`, glow `g_c`, laser `trans_q`, the
    BLINDED gate, the sensor signal) are bit-identical to the pre-#88 integer
    law on `S_old`. That law is written into the test as the documented
    expression.
  - Float readers (poison dose, recorder, `dequantize_trace_f32`) are
    float-identical.
  - *Breaks if* any reader skips or doubles its conversion, or reads a trace
    plane through a bulk name.
- **V4, writer scale.** The soot mint into an empty cell writes exactly
  `trace_from_bulk(soot)`, with the bulk N2 and `e_shed` bit-identical to
  the shift-free computation. A FieldEdit ADD of amount `a` onto a tile holding `v ≥ 0`
  leaves `quantize_trace_scalar(dequantize_trace(v) + a·w)`, i.e. `v +
  quantize_trace(a·w)`. *Breaks if* a writer mints in the bulk
  currency, or the soot split moves the bulk books.
- **V5, no mixed currency (P0).** A pump or vent extraction from a smoky tile
  removes the same bulk N as from a clear tile with equal bulk. The trace
  removed satisfies `take_t·N ≤ bulk_taken·S_t < (take_t + 1)·N`. *Breaks
  if* trace counts enter a bulk budget again.
- **V6, the books close with clip.** The §5.1 identity is exact per tick per
  plane, on the v2 scenarios (sealed blast, seal-then-breach, vent) and on a
  rail scenario: a cell seeded near `TRACE_CELL_MAX` takes a deposit, a blast
  compression and a soot mint. The cell never exceeds the rail, never goes
  negative, and `clip > 0` books exactly the excess. *Breaks if* any
  narrowing site wraps, raises, or drops without booking.
- **V7, the decay is the configured exponential where visible (P5).** A
  smoke cell from τ = 1 to τ = 0.03 per tile reaches each τ within 5 % of
  `ln(τ0/τ)/decay`, and reaches 0 in bounded time (design v2 V8's property,
  kept). Poison at `poison_min_density` decays at its configured rate within
  5 %. *Breaks if* the resolution is lost (a shift back to Q16), or the
  rounding reverts to the linear regime.
- **V8, diffusion resolves visible gradients.** In still air, a blob at
  τ = 0.3 per tile has a growing second moment within 10 s, conserves `Σ S`
  exactly, and keeps a uniform field uniform. *Breaks if* the trace currency
  is coarsened back.
- **V9, CPU == CUDA at tolerance 0**, chained and resident, on every trace
  plane and every counter including clip.
- **V10, the door.** A config giving a conservative row a non-zero
  `heat_absorb`, `absorption` or `scatter_albedo` is refused with a message
  naming #88. *Breaks if* a bulk plane could enter a trace-currency density
  sum.
- **Carried over unchanged:** design v2's V1–V9 (conservation, no expansion
  mint, venting closes, non-negativity, the air does not see the smoke, and
  the rest). Their fixtures move to `quantize_trace`.

## 11. Patch plan

The order is fixed: each patch's gate is the previous patch's state. CPU builds
use `cpp/build_cpu_home.bat` and CUDA builds `cpp/build_cuda.bat`
(`docs/dev_setup.md`, Home Desktop), run in the worktree. Python is
`C:/Users/steen/anaconda3/python.exe`; tests are `pytest tests -q`. Every patch
runs in this branch's worktree with exactly one writer at a time. No patch is
HUMAN-TEST (Erik, 2026-10-20).

| # | content | gate (oracle) | mode | tier |
|---|---|---|---|---|
| **P0** | Trace leaves the bulk split: `extract_gas_n_vec` splits over bulk planes only and prices trace by air; `extract_gas_n` returns the bulk total; `inject_gas_n_vec`'s gate over bulk. Vent/pump tests re-greened as properties. If the golden moves, re-baseline once with rationale | V5; the vent and pump suites; golden reproduced, or re-baselined with a sole-cause check | worktree subagent | Opus (vent tests may pin values; judgement on rewrites) |
| **P1** | The currency at **k = 0**: the C++ kit (§4) and the binding; the `gas_fixed` split with the deleted names, every caller moved (§2, including the bulk users); the reference FIRST (§7) with its gates re-run; the sweep / laser / readout / sensor / recorder / renderer / tools conversions; the door restriction (V10); the guard (V1). CPU and CUDA built | **V2: `GOLDEN_AGGREGATE` unchanged on CPU and every CUDA check**; full suite green; V1, V10 | worktree subagent | Opus (wide, many files; the oracle cannot see a missed conversion at k = 0, hence V3 in P3) |
| **P2** | The rail: `trace_add_rail` at every §5.3 site on CPU and CUDA; `trace_clip_sum` / `trace_clip_books` and their bindings; the Σ_map assert deleted; the Python OverflowErrors on trace slices replaced | V6 (rail scenario); golden unchanged; CUDA rail test GPU == CPU | worktree subagent | Sonnet (tol-0 and golden-unchanged oracles) |
| **P3** | **The flip: `TRACE_FINE_BITS = 8`** in the four twins. Test fixtures that wrote raw Q16 trace move to `quantize_trace` (about 50 files, §2.4's test list in the inventory). Reference gates at k = 8 non-vacuous. Spec v7. **`GOLDEN_AGGREGATE` re-baselined ONCE** with the written rationale: the fields that moved, the VERIFIED-SOLE-CAUSE paragraph (the same tree at 0 reproduces the previous value), and reproduced twice | V3, V4, V8; v2's V1–V9; full CPU suite | worktree subagent | Opus (judgement on ~50 test rewrites: properties, never bent to green) |
| **P4** | CUDA lockstep at k = 8: every `GOLDEN_AGGREGATE` CUDA consumer re-greened, chained and resident | V9 | worktree subagent | Sonnet (tol-0 oracle) |
| **P5** | Decay rounding amendment in `trace_decay.h` (one edit, both backends): `lost = min(v, max(1, (v·frac_q + 2^15) >> 16))`. The header comment and design v2 §2.3 / D4 cross-referenced. A second golden re-baseline with its own sole cause (ceil restored reproduces P3's value) | V7; v2's V8; CPU suite; CUDA trace check at tol 0 | worktree subagent | Sonnet |
| **P6** | Close: the CLAUDE.md rows (below), survey row, the smoke-tuning skill's §6 dated line, the #88 comment, the Roadmap #46 order, merge `--no-ff` into fire-12 on green | suite green | inline | — |

Critics run one at a time before P0, each verdict written into §14 before the
next spawns, on Opus.

## 12. Systems

**(a) Existing canonical systems this design uses:**
- **Trace transport** (stage 3b/3c, the tail, `trace_decay.h`): law unchanged,
  narrow railed, decay rounding amended.
- **Fixed-point kit** (`fixed_point.h` / `cuda_fixedpoint_device.cuh`): the
  trace currency lives there.
- **Q16 boundary modules**: `gas_fixed.py` splits into bulk and trace names.
- **FieldEdit**: the one trace deposit door; integer add and the rail.
- **Gas pump primitives**, **vent / duct system**: P0 removes the mixed split.
- **Material / gas tables**: the door's new bulk-coefficient refusal.
- **Radiation sweep**, **extinction planes** (the gas term), the **integer
  reference** (changed first), and the **light accessor** (reads unchanged
  floats).
- **Recorder** (contract unchanged: real-unit floats).
- **Field digest / GOLDEN_AGGREGATE** (v7, one re-baseline per approved
  change), the **CUDA harness**, the **tile inspector**, and the **sensor
  accessor**.

**(b) New, each with a draft CLAUDE.md rule:**
- **Trace currency** (`fixed_point.h` `TRACE_FINE_BITS` + the kit, twin
  `gas_fixed.py` `trace_*`). Draft rule: "The five trace planes are int32 in
  the TRACE currency, `2^(16+TRACE_FINE_BITS)` counts per real unit (#88);
  the bulk planes in the same `gas` array stay Q16. Every trace read or write
  converts ONCE through the kit (`trace_from_bulk`, `trace_density_shr`,
  `quantize_trace`, `trace_add_rail`) or `gas_fixed.trace_*`, never an inline
  shift and never a bulk name. Coefficients stay Q16 per real unit. A sum
  never mixes the two currencies: trace is priced by air, and the gases door
  refuses an optical coefficient on a bulk row. The twins and the digest
  spec's `trace_fine_bits` are held equal by a test."
- **The trace rail** (`trace_add_rail`, books `EOSSolver.trace_clip_sum` and
  `GameMap.trace_clip_books`). Folded into the Trace transport row as its
  amendment: "...No trace plane is clamped anywhere outside a reader, bar the
  counted int32 rail `TRACE_CELL_MAX` at every site that adds to a cell
  (deposits, 3b narrow, soot mint, W3/W5, seal/inject), booked in `clip`.
  Every loss is booked: vent, wipe, sink, decay, clip."
- **Amended rows**:
  - Extinction planes: "the density sum is in the trace currency, finished
    by `trace_density_shr`".
  - Integer reference: names `TRACE_FINE_BITS`.

## 13. ACCEPTED GAPs and decisions

- **ACCEPTED GAP G1, the decay fraction's own resolution.** `frac_q =
  quantize(decay·dt)` in Q16 is off by +0.7 % for smoke (22 vs 21.85), poison
  and steam, −1.1 % for teargas and −2.3 % for fuel_gas. A finer fraction
  (Q24) would fix it, but it is a different change to the law. Recorded, not
  fixed (one system at a time).
- **ACCEPTED GAP G2, the faint tail is linear.** Below τ ≈ 0.01 per tile (1 %
  opacity) decay removes 1 count a tick, so the last haze leaves in about 52 s
  instead of never. This is the D4 bounded-time property, kept on purpose.
- **ACCEPTED GAP G3, no bulk absorber.** The door refuses optical
  coefficients on the bulk rows, so a future "burnt products absorb" row on
  inert_N2 (already an ACCEPTED GAP of P5c) needs a second, bulk-currency
  partial sum in the sweep and the laser.
- **ACCEPTED GAP G4, the rail binds under sustained abuse in a closet.** A
  Miasma Vent held in a 3×3 room reaches about 230 units per cell at
  equilibrium. The rail books the excess, and the dose is already lethal there.
- **ACCEPTED GAP G5, the soot mint is not finer.** Soot is minted in bulk
  counts and shifted (`soot << 8`), so the mint's resolution is unchanged. A
  finer mint would break the exact Dalton split with N2 and `e_shed`.
- **Decision D-A (P0):** trace is priced by air on extraction, which changes
  vent and pump behaviour in smoke slightly. It is required: without it the
  rescale would let trace eat 256 times more of a bulk budget.
- **Decision D-B:** diffusion becoming live at visible densities is accepted as
  part of the rescale, not split off. It is the same law and dials, resolved.
  Design v2 D6 ("diffusion stays, with today's dials") assumed a diffusion it
  never measured working.
- **Decision D-C (P5):** the decay rounding is amended alongside the rescale.
  It is the issue's own goal (Erik's 2026-10-05 ruling), it is one edit in the
  header made for it, and it gets its own re-baseline.

**Nothing here needs Erik** under his delegation. The one call that touches
his earlier ruling is the rail beside D1. It is stated in §5.3 and §12 for him
to read. It does not cut any value the game can show.

## 14. Critique rounds

| lens | verdict | blockers |
|---|---|---|
| arithmetic, overflow, determinism (Opus, 2026-10-20) | PASS WITH FIXES | none. 3 MAJOR to fix before P1/P3 (A1 the teargas gate is not reader-invariant, so V3 as written fails; A2 numpy 1.26 on Windows accumulates int32 sums in int32; A3 `quantize_trace` narrows to int32 before the rail can see the excess). 9 MINOR. See §15 |

## 15. Critique — arithmetic lens

Scope: every arithmetic site that touches a trace value, on CPU, chained CUDA and
resident CUDA, read in the code at `3d83503`. Grep over `cpp/src` for every loop over
`n_gases`: every C++ N-total sum is gated on `gas_conservative` (`eos_solver.cpp:294,
570, 937, 2327`; `physics_engine.cpp:186, 644`; `cuda_eos_step.cu:249, 667`;
`cuda_eos_resident.cu:146`; `cuda_kick_compression.cu:706`; `sky_exchange.cpp` is
bulk only). So §2.2's mixed-currency list is complete on the C++ side. On the Python
side, the only mixed sites are the ones §2.2 already names.

**Confirmed sound (no finding):**
- **Sweep heat, light and glow terms** (`radiation_sweep.h:242-277`, CPU
  `radiation_sweep.cpp:600-676`, CUDA `cuda_radiation_sweep.cu:262-318`, binding
  `bindings.cpp:2672-2674`). They are `floor(Σ c·max(0,S) / 2^16)` with every term
  ≥ 0, so `(Σ c·S·2^k) >> (16+k) == (Σ c·S) >> 16` exactly, including at the ONE cap
  and in `exp_neg_q16`'s argument. Fleck's gas arm reads only the capped `a_gas`.
  The int64 headroom is unchanged (2^28 · 2^31 per term).
- **Laser** (`combat.py:1153-1157`). It floors each term, Python ints, and skips
  `density_q > 0`, so it is invariant term by term.
- **Poison dose** (`exchange.py:681`). It divides by a power of two in float64, so
  it is exact. The poison threshold is invariant **by luck** (see A1).
- **Recorder** (`recorder.py:190-194`), **render** and **`gas_fixed.dequantize_f32`**.
  float64 `/2^k` and then a float32 cast commutes with the power-of-two rescale, so
  the floats are identical.
- **Stage 3b `price_face`** (`bulk_transport.cpp:278-282`, CUDA
  `cuda_bulk_transport.cu:229`). The products are `dq·q ≤ S < 2^31` (since
  `dq ≤ N`) and `dq·r < N²`. No new overflow at k = 8. §2.5's
  "`dq·S < 2^61`" describes a product the code never forms; the real bound is
  better.
- **Diffusion** (`bulk_transport.cpp:752-760`, resident `cuda_bulk_transport.cu:817-825`).
  `c ≤ 2^14` and `|ΔS| < 2^31`, so the product is `< 2^45`. The magnitude-truncated
  Jacobi gives `s_new ≤ max(neighbours)`, so the claim that no rail is needed holds.
- **Decay** (`trace_decay.h:42-47`). The product is `< 2^47`. The P5 formula is
  correct at `frac_q = ONE`.
- **Decay table §3.2, §1 thresholds and §3.3 ceilings.** I re-derived them
  independently: every row of §3.2 reproduces to the second.
- **Soot mint** (`combustion.cpp:1003-1005`, `cuda_combustion.cu:518-520`). It writes
  its own cell with no atomics, so a per-cell rail is order-free on the device.
- **CUDA books.** The `unsigned long long` `atomicAdd` of signed values is
  two's-complement modular, so it is order-free. A clip slot in the same idiom is fine.
- **The `GOLDEN_AGGREGATE` arithmetic.** `quantize(0.6)·256 = 10066432 ≠ 10066330`
  checks out.

### A1. MAJOR: the teargas BLINDED gate is NOT reader-invariant, so V3 fails as specified

- **Evidence.** `exchange.py:625, 631` compares `density_q < quantize_scalar(0.15)`.
  - Old threshold: `round(0.15·2^16) = 9830`.
  - New threshold: `round(0.15·2^24) = 2516582`, while `9830·256 = 2516480`.
  - At `S_old = 9830` the old gate blinds (9830 ≥ 9830), but the new gate does not
    (2516480 < 2516582).
- **Why poison escapes.** Poison (`poison_min_density = 0.05`) is invariant only
  because its old quantization rounded *up* (3276.8 → 3277).
- **The general rule.** Any threshold quantized at the finer currency moves the gate
  by up to one old count when the old quantization rounded down.
- **Consequence.** §6 and V3 list "the BLINDED gate" as bit-identical on `S_old << k`.
  The V3 test would fail on teargas, and the next implementer would be tempted to
  bend it.
- **Fix.** Add a row to §6's "allowed to differ" table: "door thresholds
  (`teargas_blind_density`, `poison_min_density`) quantize at 2^24: the gate moves by
  less than one old count, toward the real-unit value". Take both gates out of V3's
  bit-identical list. In their place, V3 asserts the real-unit law:
  `gate(S) ⇔ S ≥ quantize_trace(th)`, and the gate is monotone in S.
  (The alternative, `quantize_bulk(th) << k`, keeps the old coarse threshold for no
  reason.)

### A2. MAJOR: "int32 `.sum()` accumulates in int64" holds only on numpy 2, and the repo documents a numpy 1.26.4 environment

- **Evidence.**
  - §2.1 rests on "numpy is 2.4.6 on this machine".
  - `docs/lenovo_dev_setup.md:147` records numpy 1.26.4 runs.
  - On Windows, numpy < 2 reduces int32 in the platform `long`, which is 32-bit.
- **Consequence.** At k = 8, one unit is 2^24, so a plane total above 128 units
  wraps. The miasma room in §3.1 is 2062 units, which is 2^35 counts. The V6 books
  identity, the §3.1 measurements and existing checks such as
  `tests/cuda_trace_smoke_check.py:297` (a bare `int(gg.gas[SMOKE][...].sum())`) would
  silently wrap on that environment.
- **The same hazard on assignment.** numpy 1.26 wraps an out-of-range Python int
  assigned into int32 with only a DeprecationWarning, where numpy 2 raises.
- **Fix.** P3 lints for any `.sum()` over a trace plane without an explicit int64
  accumulator, using the repo's own idiom `.astype(np.int64).sum()` (as
  `gamemap.py:2295` and `test_a5_seal_evacuation.py:62` already do). This replaces
  §2.1's "grep for `dtype=np.int32`". Every Python rail returns a Python int that is
  range-checked before it is stored.

### A3. MAJOR: `quantize_trace` narrows before the rail can see the excess, and the C++ `quantize` it copies does not saturate

- **Evidence, C++.** `fixed_point.h:92-95` is
  `quantize(double v) { return (q16)(scaled ± 0.5); }`. This is a direct
  double → int32 cast with no saturation, which is UB out of range. At k = 8,
  "out of range" drops from 32768 units to 128 units.
- **Evidence, Python.** `gas_fixed.py:43-52` ends in `out.astype(np.int32)`, which
  wraps silently (or gives INT32_MIN) for values of 128 units or more.
- **Where this bites.**
  - §4 types `quantize_trace(double)` as "the door-2 idiom matching `quantize`'s",
    and §5.3 promises a rail at every adding site.
  - A W5 puff, or an array seed (`tools/lighting_demo.py:916-930`, test fixtures)
    of 128 units or more, wraps inside the conversion before `trace_add_rail` runs.
  - The same applies to `trace_from_bulk(q) = q << 8` on a negative `int64`, which is
    UB before C++20. It is safe today because `soot ≥ 0`, but the kit should not
    rely on that.
- **Fix.**
  - `quantize_trace` returns **int64** (C++) or a Python int / int64 array (Python),
    never int32.
  - `trace_from_bulk` is written as `q * (1 << TRACE_FINE_BITS)`, with
    `assert(q >= 0)`.
  - Only `trace_add_rail` narrows to int32.
  - V6 adds one case: a W5 puff, or a FieldEdit, of 200 units into an empty cell
    lands at `TRACE_CELL_MAX` with exactly the excess in `clip`.

### A4. MINOR: the rail's negative-result semantics are unspecified

- "It never produces a negative" can be read as "clamps a negative result to 0".
  That would mint mass without booking it.
- At stage 3b, `ds` is a net and can be negative. `s + ds ≥ 0` holds by the limiter,
  and today it is asserted (`bulk_transport.cpp:542`).
- **Fix.** Specify that the rail clamps only from above. A negative `s + add` is a
  contract violation: keep the `assert` (CPU), and on CUDA count it in a
  diagnostic slot rather than clamp it. Apply the rail once per cell, on the net
  `s_own + ds`, never per face.

### A5. MINOR: P0's priced-by-air extraction needs an N_EPS guard

- **Evidence.** `extract_gas_n_vec` (`gamemap.py:3096-3099`) today returns zeros
  when the all-planes `n_total ≤ 0`.
- With the split over bulk only, a tile with `N_bulk == 0` and `S > 0` makes
  `take_t = floordiv(bulk_taken·S_t, N_bulk)` divide by zero. Such a tile is
  possible: a deposit before the tail's sink or 3c wipe runs.
- **Fix.** Guard with `N_bulk < N_EPS_RAW → take nothing` (the 3b donor guard), and
  add the case to V5.

### A6. MINOR: do not change `inject_gas_n_vec`'s `> 0` gate to bulk only

- `gamemap.py:3151-3152` gates on `sum(add_vec) > 0`. Every slice is ≥ 0, so the gate
  is already currency-free.
- Gating on bulk alone would silently skip a trace-only `add_vec`, a mass loss with
  no book. §2.2 marks the gate "harmless", and P0's row says "`inject_gas_n_vec`'s
  gate over bulk".
- **Fix.** Leave the gate as it is.

### A7. MINOR: the sensor conversion must sit where both `sample` and `area` see it

- `sensor_accessor.py:146-175`: `sample` returns the array cell, and `area` returns a
  raw `(sum, n)` that `sensor_system.py:92` floors as `total // n`.
- Converting only the scalar return leaves area sensors 256 times off.
- **Fix.** Apply `trace_to_q16` in `_channel_array` for `Channel.SMOKE` (an array
  `>> TRACE_FINE_BITS`). Both paths then read Q16. This is invariant on
  `S_old << k`.

### A8. MINOR: the reference gates and gate 0 never see a fine count

- `sweep_ref_q_gates.py:104-106` draws smoke and steam from a fixed set of Q16
  values. "`trace_from_bulk` of today's draw" makes every drawn density a multiple
  of 256.
- Gate 0 (C++ == reference) then cannot tell a `>> 24` from a rounding variant on
  non-multiples, nor test the shift's low bits.
- **Fix.** Draw `trace_from_bulk(v) + rng.randrange(256)` on non-zero trace cells.
  Shift **only the trace planes** (`smoke`, `steam`); the bulk pair and `n_bulk`
  stay bulk.

### A9. MINOR: §6 contradicts itself on `gas_energy`

- Line ~395 says `temperature` and `gas_energy` move through `a_gas`. Line ~444 says
  "Bulk N, wind and `gas_energy` stay bit-identical with or without trace".
- Since P5c the gas fold lands smoke-absorbed radiation in `gas_energy`, so the
  second statement is false whenever smoke absorbs.
- **Fix.** Restrict v2's V5 to "bulk N and wind; `gas_energy` too when the sweep's
  gas term is off (`heat_absorb = 0`)".

### A10. MINOR: the headroom claim is weaker than §3.4 states

- §3.4 says k = 8 "keeps the rail out of every measured legal case". That is true of
  each row alone, but two measured legal effects compose: §3.1's grenade spam
  (10.95 units) under §3.1's own 18× blast compression gives about 197 units, above
  the 128 ceiling. G4 already admits the closet case.
- This does not change the decision, since k = 9 is worse. But the rail should be
  presented as **expected to bind in reachable abuse**, not as "zero in every
  measured legal scenario" (§5.1). V6's rail scenario is therefore a primary gate,
  not a corner case.

### A11. MINOR: arithmetic slips in the text

- §4 point 1: `0.22·2^16/256 = 56.32 → 56` is a 0.57 % rounding, not 1.8 %. The
  argument still holds.
- §2.5's `dq·S` bound: see "Confirmed sound" above.
- V4's identity `quantize_trace_scalar(dequantize_trace(v) + x) == v + quantize_trace(x)`
  is not a theorem in float64: a tie within about 2^-21 counts of .5 can differ.
  State it as a property within one count, or draw `x` off ties.

### A12. MINOR: name the Q16 fractions that are neither bulk nor trace

- §4 offers only `bulk_*` and `trace_*` names, but `gas_fixed.quantize_scalar` also
  quantizes Q16 *fractions and temperatures*: filter efficiencies
  (`vent_system.py:121-122`, validated against `FP_ONE`), `t_min` / `t_max`
  (`:371`), the sky λ, and the O2 fraction.
- A filter efficiency is "per trace gas", so an implementer may move it to
  `quantize_trace`. That gives `eff_q` up to 2^24, `scrub = (r·eff_q) >> 16` then
  scrubs 256 times the removed mass, and the validator, renamed to `TRACE_ONE`,
  would pass it.
- **Fix.** Add a `quantize_q16_scalar` / `Q16_ONE` name for fractions (or point those
  callers at the shared fixed-point kit). V1's lint asserts that the filter validator
  bounds against `Q16_ONE`.
- Also: `unseal_tiles` needs no rail. `target = Σ avail // (k+1) < max(avail)`
  (`gamemap.py:2898`), so §5.3's row can drop it.
