# Smoke-unit rescale: a finer trace currency (design v2, 2026-10-20)

**Issue:** #88. **Branch:** `88-smoke-unit-rescale`, cut off `fire-12` at `c8cf9da`.
**Status:** v2. Three critique lenses have been run (§15 arithmetic, §16 books
and semantics, §17 inventory and plan), all PASS WITH FIXES with no blockers.
Every finding is resolved below (§14.1 maps each one to where); the orchestrator
ruled B1–B4 on 2026-10-20. Erik delegated #88 to Claude end to end on 2026-10-20
("you can design it as well as I, and it doesn't need any human test").
**Out of scope** (Erik's word): the soot AMOUNT (`soot_yield`, the studio's
2–4 m visibility), revisited after #88.
**Depends on:** `docs/smoke_transport_design_2026-10-04.md` (v2 transport,
D1–D6) and `cpp/src/trace_decay.h`.

**Changelog v1 → v2.**
- **Books.** The deposit and mint books are added, written at the attempted
  amount before the saturation, so the identity closes with no residual (B1).
  The pump, FieldEdit REMOVE and the explosion clear stay unbooked, as a
  labelled ACCEPTED GAP G6 with CLAUDE.md's false "already-booked" corrected in
  P6 (B2, orchestrator's ruling).
- **The saturation.** The rail is renamed: the counted int32 saturation, kit
  `trace_add_sat`, books `trace_clip_*` (C7). It is described as what it is,
  D1's option (b) at the representability limit (B3). It is expected to bind in
  reachable abuse (A10), saturates from above only (A4), and its every site is
  gated per backend (C10).
- **Conversions.** `quantize_trace` returns int64 and only the saturation
  narrows (A3). A third name, `q16_*`, covers Q16 fractions and temperatures
  (A12). Thresholds move by less than one old count (A1). The sensor converts in
  `_channel_array` (A7). The renderer gets one trace-dequantize helper (C9).
  int64 accumulators are linted (A2).
- **Verification.** V3 drives public entry points through a reader registry
  (C2). A fixture inventory is committed (§2.7, C1), with a write lint. P1 gets
  a "dry flip" at k = 8 (C3).
- **Plan.** ONE golden event for the whole branch: the flip (k 0→8), the decay
  rounding (ceil→nearest) and CUDA land as one gated patch P3, with a
  three-stage sole-cause proof (B4, C11). P0 may not move the golden (B4).
  C4's build discipline is a MUST in every C++ patch. The patches are re-tiered.
- **Text.** An intended-behaviour-changes table is added (§6.4, B5), plus the
  smaller corrections A5, A6, A8, A9, A11, B6–B9, C5, C6, C8, C12 and C13.

## 0. Summary

- **The rescale.** The five trace planes (steam, smoke, poison, teargas,
  fuel_gas) get **`TRACE_FINE_BITS = 8`** more fractional bits: one unit of
  trace density becomes `2^24` counts instead of `2^16`. Unchanged: the int32
  storage, the transport law, every config value, and every reader's output in
  real units. Only the integer currency of the trace planes changes. The bulk
  planes (O2, inert_N2) keep Q16.16.
- **Why 8.** It is the measured compromise (§3). It gives 128 units of
  per-cell headroom: 440 times the largest real per-cell value, and 3.3 times
  the worst single measured abuse. With the nearest-rounded decay, the
  configured exponential holds down to the edge of visibility.
- **What the rescale exposes**, each fixed in its own place:
  - the pump/vent extraction split mixes trace counts with bulk counts (P0);
  - an int32 narrow raises or silently wraps once a cell can reach `INT32_MAX`
    (P2: the counted saturation, with the mint and deposit books that let its
    identity close);
  - the ceil decay rounding biases the faint end (amended to nearest with a
    floor of 1, inside the flip, P3).
- **One constant, one owner per language, twins held equal by a test.** The
  ambiguous `gas_fixed` names are **deleted** and replaced by `bulk_*`,
  `trace_*` and `q16_*` names, so a missed Python site fails loudly. Two lints
  catch the silent classes: a write lint on trace slices and a no-bulk-name-on-
  trace lint.
- **ONE golden event, with a three-stage sole-cause proof** (§6.2):
  1. P1/P2's tree at `TRACE_FINE_BITS = 0` with ceil decay reproduces today's
     `GOLDEN_AGGREGATE` (`f6ba810e…`) bit for bit, on CPU and CUDA;
  2. the flip to 8 with ceil gives an intermediate value X, recorded and never
     committed as the golden;
  3. nearest rounding gives the committed Y.

## 1. The problem, restated with numbers

One Q16 count of smoke is τ = 134.29 / 65536 ≈ 0.0020 per tile (green,
`[gases.smoke] light_absorb`). So "faintly visible" (τ ≈ 0.01) to "black"
(τ ≈ 4) is 5 to 2000 counts: visible smoke lives in the bottom ~11 bits of
the int32 plane. At that resolution, three laws break:

| law | integer form | threshold at Q16 | effect at Q16 |
|---|---|---|---|
| decay | `lost = ceil(v·frac_q / 2^16)`, `frac_q = 22` (0.008/s at 24 tps) | exponential needs `v ≫ 2979` counts, i.e. τ ≫ 6 | every visible tile loses 1 count a tick: a straight line to zero. Opaque, then gone at once (Erik's report) |
| diffusion | `|F| = (c·|ΔS|) >> 16`, `c = dd_q = quantize(0.10/24) = 273` | `|ΔS| ≥ 240` counts, i.e. Δτ ≥ 0.49 per tile | still-air spreading of smoke is DEAD across almost the whole visible range |
| poison near its threshold | the same decay | `poison_min_density = 0.05` = 3277 counts, under poison's own exponential threshold (5958) | poison at the dose edge decays linearly, about 1.7 times its configured rate (§6.4) |

Erik's ruling (2026-10-05): the configured decay (`[gases.smoke] decay`, a
125 s e-fold) is the rate the game must actually show.

## 2. Inventory: every writer and reader of the trace planes

Collected by grep over the worktree, 2026-10-20, and re-swept independently by
lens 3 (§17, "confirmed sound": the C++ list is complete; no saves exist; old
recordings stay valid). "After #88" says what the site must do. **INLINE**
marks a site that today applies `>> 16`, `/ 65536` or `FP_ONE` to a trace
value. After #88 every such site goes through a kit function (§4).

### 2.1 Cross-cutting facts

- **Two currencies in one array.** `GameMap.gas` is one `(N_GASES, h, w)`
  int32 array (`gamemap.py:360`): trace ids 0..4, bulk ids 5..6.
  `gmap.smoke` is a view (`:381`). After #88 the array holds two currencies.
  That is the main hazard, so every site that treats all seven planes as one
  unit is listed in §2.2, and removed or guarded before the flip.
- **`gas_fixed.py` has three kinds of user.** Its constants and functions
  serve:
  - the trace planes;
  - the bulk planes: O2/N2 seeding (`gamemap.py:873-885`, `:1384-1395`,
    `:3040-3045`), `field_edit._combine_atmosphere_to_N` (`:463-473`), the
    pump ΔN (`pump_system.py:142-147`), the vent rate (`vent_system.py:322`),
    `bulk_n` in the readout (`hover_readout.py:176-179`);
  - **Q16 fractions and temperatures**: filter efficiencies and their
    validator (`vent_system.py:120-128`), `t_min` / `t_max`
    (`vent_system.py:371`), the T_AMB fold (`gamemap.py:925`,
    `physics_runner.py:1835`), the sky λ (`physics_runner.py:1287`),
    `ambient.py`, `temperature_scale.py:33`, the O2 fraction.

  It splits into three names accordingly (§4, A12).
- **int64 accumulation is a requirement, not an accident.** This machine runs
  numpy 2.4.6, which reduces int32 in int64. But `docs/lenovo_dev_setup.md:147`
  records numpy 1.26.4, which on Windows reduces int32 in 32-bit `long`, and
  wraps an out-of-range Python int assigned into int32 with only a warning. At
  k = 8 a plane above 128 units total overflows int32 (the miasma room holds
  2062).
  - So **every sum over a trace plane uses an explicit int64 accumulator**,
    the repo's `.astype(np.int64).sum()` idiom (`gamemap.py:2295`,
    `test_a5_seal_evacuation.py:62`).
  - It is linted (V1, A2), in `src/`, `renderer/`, `tools/` and `tests/`
    (e.g. `cuda_trace_smoke_check.py:297`'s bare `.sum()`).
  - Every Python saturation returns a Python int, range-checked before it is
    stored.

### 2.2 Sites that MIX trace and bulk counts (removed or guarded before the flip)

| site | what it does | fix |
|---|---|---|
| `gamemap.py:3081-3115` `extract_gas_n_vec`, and its callers `extract_gas_n` (`pump_system.py:113`) and `vent_system.py:390-391` | splits a bulk budget `want` over the holdings of **all seven** planes (`gas_proportional_split`), so trace counts are spent as if they were air. At k = 8, a tile of real-max smoke (0.29 units = 4.87 M counts) on ambient air (65 536) would split a pump's budget 98.7 % trace (§16 "P0 is required") | **P0**: the split runs over the bulk planes only. Each trace plane leaves **priced by air**, `take_t = floordiv(bulk_taken · S_t, N_bulk)`, the ratio law stage 3b and the vent supply side (`vent_system.py:476`) already use, so it is scale-invariant. **Guarded by `N_bulk < N_EPS_RAW → no trace taken`** (the 3b donor guard, A5). The scalar `extract_gas_n` returns the bulk total. What the pump does with the trace it takes is unchanged: it is destroyed, unbooked, as today (G6, orchestrator's B2 ruling) |
| `gamemap.py:3151` `inject_gas_n_vec` `sum(add_vec) > 0` | a `> 0` gate. Every slice is ≥ 0, so it is already currency-free | **left as it is** (A6). A bulk-only gate would skip a trace-only injection unbooked |
| `combat.py:1131-1163` `fire_beam` | loops over all `N_GASES`, `(absorb[g]·S) >> 16` | loops over the trace ids; the shift goes through the kit (P1) |
| radiation sweep, `radiation_sweep.cpp:507-522, 595-606, 664-677` (+ CUDA `:175, 263-265, 311-318, 956-975`) | the density sums run over every gas with a non-zero coefficient | **the gases door refuses a non-zero `heat_absorb`, `absorption`-derived light/beam coefficient or `scatter_albedo` on a conservative (bulk) row** (P1). The sums then only ever see trace planes, and are in the trace currency. Every shipped bulk row is 0, so nothing ships differently (G3) |
| `hover_readout.py:161-164` | dequantizes STEAM..FUEL_GAS **and O2** through one `_GAS_FP_ONE_F` | per plane: trace ids through `dequantize_trace`, O2 through `dequantize_bulk`. `bulk_n` (`:176-179`) through `dequantize_bulk`; the `gas_energy` display's `_GAS_FP_ONE_F ** 2` becomes `BULK_ONE ** 2`, never `TRACE_ONE` (C8) |
| `hover_readout.py:241-245` | all seven planes go to `gas_extinction_q16` | unchanged: the door makes the bulk coefficients 0 |
| `src/debug_keys.py:155-168` `debug_spawn_gas` | writes `SMOKE_MAX_Q` into any gas id, bulk included | `TRACE_ONE` on a trace id, `BULK_ONE` on a bulk id |
| `gamemap.py:2662-2670` (`_seal_blockers`), `:3010-3017` (`_inject_gas_n_core`) | a `limit = 2^31` OverflowError per slice | trace slices take the counted saturation inside the raise-free mutation pass (`gamemap.py:2751-2755`, §5.3); the bulk slices keep the error |
| **test** `test_ps1_smoke_roundtrip.py:119-122` | asserts that `Σ` over ALL planes never rises. At k = 8 the soot mint takes `soot` bulk counts from N2 and adds `soot << 8` to smoke, so the total rises (C1) | rewritten in P1 as the right property: `Σ bulk` never rises, and the trace planes close by their books (V6). Never bent to green |

**Mixed but exact (no change; listed so that nobody "fixes" them, C8):**
`test_vent_mechanism.py:81-87`, `test_b4_pump.py:386-389`.

### 2.3 Writers

The saturation sites are the ones `§5.3` names.

| writer | where (CPU / CUDA) | today | after #88 |
|---|---|---|---|
| FieldEdit trace ADD / MAX / REMOVE: the door for `emit_gas`, `add_explosion_smoke`, `blast_smoke_peak`, spray cones, the `physics.py` smoke clear | `field_edit.py:499-510` `_combine_gas` (**INLINE** `/ FP_ONE_F`, `quantize_scalar`) | dequantize the tile, combine in float, re-quantize | the same combine under the mode's semantics (`dequantize_trace` → float64 → `quantize_trace_scalar`, a Python int), then the saturation. Exact in float64 for any int32 tile at 2^24. The positive change (ADD, MAX) is booked at its attempted amount in `trace_deposit_books` (§5.1); REMOVE books nothing (G6). At k = 0 this is the old code path verbatim |
| combustion soot mint | `combustion.cpp:1003-1005`, `cuda_combustion.cu:518-520` | `soot = narrow_round(burn·soot_yield_q)`, then `SOOT += soot`. `soot` is in **bulk** counts, and is also subtracted from N2 and `e_shed` | `SOOT = trace_add_sat(SOOT, trace_from_bulk(soot), &clip)`, with the attempted `trace_from_bulk(soot)` booked in `trace_mint_sum`. The Dalton split and `e_shed` keep the bulk-count `soot`, so the bulk and energy books are bit-identical and the mint is exactly the old × 2^8 |
| W5 steam puff | `physics_engine.cpp:1044-1045` (the resident path runs this CPU tail) | `puff_q = quantize(steam_yield·boiled_m)` | `quantize_trace(...)` (int64), booked in `trace_mint_sum`, then the saturation |
| W3 water evacuation | `physics_engine.cpp:1073-1140` | `evac = mul_q16(plane, frac_q)`, shared to neighbours with the remainder to the last | scale-invariant and conserving (§16). The receiver add takes the saturation |
| stage 3b advection | `bulk_transport.cpp:457-548`; `cuda_bulk_transport.cu:403-476` `trace_apply` (chained and resident) | gather; int32 narrow with `assert` (CPU) / unchecked (CUDA) | `price_face` is a ratio, so unchanged. The saturation is applied ONCE per cell, on the net `s_own + ds`, never per face (A4) |
| stage 3c wipe | `bulk_transport.cpp:619-641`; `cuda_bulk_transport.cu:483-498` | sub-N_EPS cells zeroed, booked | unchanged |
| trace tail: sink, diffusion, decay | `bulk_transport.cpp:681-809`; resident `cuda_bulk_transport.cu:834-962` | sink zeroes; Jacobi `(c·|ΔS|) >> 16`; ceil decay | `>> 16` there is a rate shift on a count, scale-invariant. Diffusion is convex (`s_new ≤ max(neighbours)`, §15), so it needs no saturation, and its asserts stay. The decay rounding is amended inside P3 |
| vents (plenum extract / supply inject) | `vent_system.py:390-520` | priced by air on supply; mixed split on extract | P0 split; the supply inject through `inject_gas_n_vec` → `_inject_gas_n_core` takes the saturation |
| `seal_tiles` redistribution | `gamemap.py:2769-2780` | divmod per slice, scale-invariant, overflow → error | the trace slices take the saturation inside the mutation pass |
| `unseal_tiles` | `gamemap.py:2896-2917` | `target = Σ avail // (k+1) < max(avail)` | **no saturation needed** (A12): it cannot exceed an existing cell |
| `destroy_wall` | `gamemap.py:2298+` | bulk only; the trace on the tile is sunk by the tail | unchanged |
| debug spawn | `src/debug_keys.py:155-168` | `= SMOKE_MAX_Q` | §2.2 |
| level load / hot reload | `gamemap.py:884-885`, `:1574-1575`, `:1793-1796` | no level authors trace gas; reload copies the array | unchanged |
| tools that seed trace | `tools/lighting_demo.py:916-930` (`SMOKE_MAX_Q·amount`), `tools/eos_p5_bake.py:68-71, 227, 285` (**INLINE** `q16()`), `tools/bench_radiation_sweep.py:127-133` (raw Q16 fixture) | Q16 | through `quantize_trace` (int64), narrowed by the saturation |

### 2.4 Readers

| reader | where | today | after #88 |
|---|---|---|---|
| sweep heat term `a_gas` | `radiation_sweep.h:248-252` `gas_extinction_finish` (**INLINE** `sum >> FP_SHIFT`). Every CPU, CUDA and binding call site goes through this one FP_HD finish (§17) | `min(ONE, Σ hq·S >> 16)` | `min(ONE, trace_density_shr(Σ hq·S))`, i.e. `>> (16 + TRACE_FINE_BITS)`. Bit-identical on `S_old·2^8`: every term is ≥ 0 and the floor commutes (§15) |
| sweep light term + glow coefficient | `radiation_sweep.h:274-277` `gas_extinction_exp_finish` (**INLINE**) and `gas_extinction_finish` for `g_c` | the same density sum `>> 16` | the same kit shift. `mul128_shr(light_q, g_c, 16)` reads the fraction `g_c`, so it is unchanged |
| integer reference | `sweep_ref_q.py:1047-1121`, `:2019` | spec | changed FIRST (§7) |
| laser | `combat.py:1157` (**INLINE**) | per-gas floored term | `trace_density_shr(absorb·S)` per term over the trace ids; bit-identical on `S_old·2^k` |
| poison dose | `exchange.py:670, 681` (**INLINE** `/ 65536.0`) | `quantize_scalar(threshold)`; `dps·S/65536·dt` | `quantize_trace_scalar(threshold)`; `dps·dequantize_trace(S)·dt`. The float is identical on `S_old·2^k`. The threshold moves by less than one old count (§6.3, A1) |
| teargas BLINDED | `exchange.py:625, 631` | `quantize_scalar(0.15)` = 9830 | `quantize_trace_scalar(0.15)` = 2516582. That is NOT `9830·256 = 2516480`: the gate moves by 0.4 old counts toward the real value (A1) |
| smoke sensor | `sensor_accessor.py:130-132`, `:146-175` (`sample`, `area`) → `sensor_system.py:86-92` → decider `KIND_Q16` thresholds (`entities/nodes.py:94`) | raw Q16 counts | `_channel_array` returns `trace_to_q16(gas[SMOKE])` (an array `>> TRACE_FINE_BITS`, floor), so `sample` and `area` both read Q16 (A7). The signal contract and authored thresholds keep their meaning |
| recorder | `recorder.py:190-194` (**INLINE** `/ 65536.0`) | real units | `dequantize_trace_f32`. The frozen `.npz` contract and old recordings are unchanged |
| renderer | `game_renderer.py:479, 486, 510, 578-579` | float units | ONE helper, `_trace_floats(gmap)` (the `(N_TRACE, h, w)` float32 block through `dequantize_trace_f32`) at `upload_state`, which the four sites slice (C9). It is V3's renderer entry point and the place a resident gather later swaps in |
| `gas_medium.py:101-104` | `light_absorb_q16 / 65536` | a COEFFICIENT | unchanged: coefficients stay Q16 per real unit |
| hover readout | §2.2 | | |
| books | `bindings.cpp:2966-2993` | counts | **counts in each plane's own currency**, documented on the binding. No consumer sums across gases (§16) |
| vent entity digest | `entities/vents.py:117-120` (`<q`, `serialize.py:88-95`) | trace counts | values ×2^8; ENTITY_SECT bytes move, format unchanged |
| tools / benches | **INLINE**: `tools/bench_clamp_shave.py:460`, `tools/storm_ledger.py:132-145`, `tools/eos_p5_bake.py:237, 292, 399, 421`, `tools/fire_timing_harness.py:98, 504-506` (local `FP_ONE`, a `mass > 1.0` raw-count threshold); via `gas_fixed`: `tools/wind_drift_probe.py:78`; coefficient only: `tools/smoke_tau_histogram.py:63`; bulk / scale-free: `tools/plot_sealedbox_70.py:40`, `tools/sky_exchange_bench.py:44`, `tools/bench_two_room.py:96-97` | Q16 | converted through the kit. Coefficient-only and scale-free sites are unchanged |
| not readers (checked) | `fire_simulation.cpp:95-98` / `cuda_fire.cu:356` (smoke `(void)`; the `bindings.cpp:1037` comment is stale), `temperature_solver`, `sky_exchange.cpp`, `water_solver`, `light_field.py`, `status.py`, `vision.py`, `ai_zombie.py` | — | — |

### 2.5 Scale-free sites (confirmed, no change)

- **`price_face(dq, S, N) = dq·floordiv(S,N) + floordiv(dq·r, N)`.** The
  products are `dq·q ≤ S < 2^31` and `dq·r < N²` (A11: v1's "`dq·S < 2^61`"
  described a product the code never forms).
- **Rates.** `trace_diffusion_dd_q` and its stability door;
  `trace_decay::frac_q`.
- **Fractions of a count.** The vent filter scrub `(r·filter_eff_q) >> 16`;
  W3 `mul_q16(plane, frac_q)`.
- **Priced by air.** `vent_system.py:476`.
- **Divmod.** `seal_tiles`.
- **Non-zero tests.** `gamemap.py:2644`.

### 2.6 Config keys denominated in trace density

All of these are REAL units, unchanged by #88, and converted at the doors
above:
- every `gas_amount`;
- `[payloads.*] soot_yield` (g/g) through `smoke_units_of_soot`;
- `[physics.combustion] soot_yield`;
- `[physics.water] steam_yield`;
- `teargas_blind_density`, `poison_min_density`, `poison_dps`;
- the `[gases.*]` `absorption` / `heat_absorb` / `light_absorb` /
  `scatter_albedo` columns, which are per real unit.

So `soot_g_per_smoke_unit`, `smoke_units_of_soot`, `test_gas_heat_absorb.py`
and `test_gas_light_absorb.py` do **not** change.

The issue's list of "everything denominated in smoke units" was written for a
rescale of the UNIT (×100/×1000 changes its meaning). A power-of-two rescale
of the CURRENCY leaves the unit's meaning alone, which is why the factor is a
power of two.

### 2.7 Test and tool fixture inventory (C1)

**How it was produced.** A grep over `tests/*.py` and `tools/*.py` (the
scratchpad's `fixtures.txt`) for every assignment or `+=` into a trace slice
(`gas[<trace id or name>]`, `.smoke`, `gas[:N_TRACE_GASES]`), classified by
its right-hand side. It found 150 lines in 47 files; the false positives are
dropped below. Every fixture moves in **P1, at k = 0**, where the move is an
identity, so a test whose outcome changes at k = 0 has been rewritten wrongly
(C1's oracle). The standing write lint (V1) then keeps the list closed.

The classes, by how they fail at k = 8 if left alone:

| class | how it fails | conversion |
|---|---|---|
| **N, name** | a deleted `gas_fixed` name: it fails LOUDLY at P1 | `quantize_trace*` / `TRACE_ONE` |
| **Q, local literal** | **SILENT**: 256× thinner | `TRACE_ONE` / `quantize_trace*` |
| **W, wrong module** | **SILENT** | `quantize_trace*` |
| **H, derived from a trace-currency quantity** (a threshold, a coefficient shift, a helper) | follows its source; **SILENT** if the source is a literal | convert the source |
| **R, raw counts for a scale-free property** (conservation, divmod, splits, digests) | the property holds at any scale | **left raw, with a one-line `# raw trace counts: scale-free (#88)` note**. A raw count meant as a DENSITY moves to `quantize_trace` |

The list:

| file:line | class | what to do |
|---|---|---|
| `tests/cuda_trace_smoke_check.py:69` (`Q = 65536`), `:130-135, 144-145, 163-165, 176-177, 191-193`; its `.sum()` at `:297` | Q | `TRACE_ONE`; int64 accumulator |
| `tests/test_smoke_transport_v2.py:48` (`Q = 65536`), `:117, 147, 222-223, 245, 282-283, 315, 351, 377, 478, 486` | Q | `TRACE_ONE` |
| `tests/test_smoke_transport_v2.py:378, 415, 452-453` | R / H | `:378` (uniform thin field) R; `:415` the V8 loop is re-bounded from the law (B6); `:452-453` follow `q` |
| `tests/test_smoke_transport_v2.py:181` (`n_bulk >> 2`) | H (a ratio from bulk) | `trace_from_bulk(n_bulk >> 2)`, ratio 1/4 |
| `tests/test_multigas_structure.py:225` (`= 65536`), `:262` (`* 65536 * 1.5`) | Q | `TRACE_ONE` |
| `tests/test_multigas_structure.py:109, 113, 117, 224, 226, 263-264, 327` | R (identity / view checks) and N (`:327`) | R kept; `:327` → `quantize_trace_scalar` |
| `tests/test_light_field.py:71` (`fire_fixed.quantize_scalar(0.4)`) | W | `quantize_trace_scalar` |
| `tests/field_ab_harness.py:136`, `tests/test_stamp_units_cpp_ab.py:76`, `tests/cool_shift_axis_gate_a_capture.py:84`, `tests/cuda_thermal_mass_gate_a_capture.py:66`, `tests/test_eos_p4_combustion.py:282`, `tests/test_hover_readout.py:372`, `tests/test_temperature_gas_radiation.py:414, 457, 745`, `tests/test_water_displacement.py:320` (`SMOKE_MAX_Q`) | N | `quantize_trace_scalar` / `TRACE_ONE` |
| `tests/cuda_p6d_light_transport_check.py:75-76`, `tests/cuda_radiation_sweep_check.py:1271-1272`, `tests/test_light_transport_setting.py:122-123`, `tests/test_radiation_sweep_gas_fleck.py:162-163`, `tests/test_radiation_sweep_smoke_wiring.py:119-120` (`dens = gas_fixed.quantize(...)`) | N | `quantize_trace` |
| `tests/_radiation_sweep_harness.py:114-132` `random_gas` (`Q(0.27)`, `ONE`). Users: gate 0 `test_radiation_sweep_reference.py`, `cuda_radiation_sweep_check.py`, `_cones_check`, `_light_check`, `test_radiation_sweep_light.py`, `test_radiation_sweep_cones_sky.py`, `test_sweep_fine_currency.py` | Q (a second reference vocabulary) | the A8 treatment: `trace_from_bulk(v) + rng.randrange(256)` on non-zero trace cells, bulk pair untouched. Every user prints its count of absorbing cells, which must be > 0 (§7) |
| `tests/test_hover_readout.py:110-114` (`round(x * GAS_FP_ONE_F)`) | H (a local alias) | `quantize_trace_scalar`; O2 rows via `quantize_bulk_scalar` |
| `tests/test_field_edit.py:49` `_smoke_q` and its users `:131, 139, 255`; the float reference `:479-496` | H | `_smoke_q` → `quantize_trace_scalar`; the float reference in real units is unchanged |
| `tests/test_laser.py:141, 174` (`density_q` derived as `(32768 << 16) // absorb_q`) | H (a coefficient shift) | the shift becomes `TRACE_SHIFT` |
| `tests/test_gas_weapons.py:161-170, 305-317` (`thresh_q ± 1`) | H (a threshold) | `thresh_q = quantize_trace_scalar(...)`: the A1 property (gate ⇔ S ≥ threshold) |
| `tests/test_b1_signal_bus.py:407`, `tests/test_b3_sensors.py:152` (`= 4242`) | R as a density → sensor | they meant a Q16 signal: seed `4242 << TRACE_FINE_BITS` via `trace_from_bulk` (A7) |
| `tests/test_a5_seal_evacuation.py:113-115, 186, 203-205, 386-387, 435` | R (divmod) | kept raw |
| `tests/test_vent_mechanism.py:136-138, 183-184, 317-318, 341`, `tests/test_vent_determinism_and_serialize.py:70-72, 128`, `tests/test_vent_guards_and_load.py:92, 160, 199, 222` | R (splits, scrub, serialization) | kept raw, re-checked against P0's new split |
| `tests/test_b4_pump.py:95, 143, 173` | R | re-checked against P0 |
| `tests/test_eos_p1_species_transport.py:149` | R | kept raw |
| `tests/test_explosion_smoke_noise.py:76`, `tests/test_destroy_wall_conserves_mass.py:101`, `tools/repro_destroy_wall_mint.py:84`, `tools/velocity_clamp_pv2_measure.py:79` | R (zeroing / scaling every plane) | kept |
| `tests/test_entity_digest.py:383`, `tests/test_fire_feedback.py:129`, `tests/test_recorder_dump.py:34`, `tests/test_field_edit.py:88` (fake maps' `self.smoke = zeros`), `tests/test_renderer_smoke.py:40, 49` (a float plane) | R | kept |
| `tools/bench_radiation_sweep.py:127` (`rng.choice([1966, 6554, 17695, ONE])`) | Q | `trace_from_bulk(...)` |
| `tools/lighting_demo.py:916-930`, `tools/eos_p5_bake.py`, `tools/fire_timing_harness.py:98, 504-506`, `tools/bench_clamp_shave.py:460`, `tools/storm_ledger.py:145`, `tools/wind_drift_probe.py:78` | Q / N | §2.3, §2.4 |
| **false positives** (local bulk arrays named `gas`): `tests/cuda_bulk_flux_check.py:240-242`, `tests/cuda_kick_check.py:322-326`, `tests/cuda_kick_drag2_timing_check.py:133-134`, `tests/test_drag2_*`, `tests/test_p_e3_drag.py`, `tests/test_velocity_clamp_property.py:105-106` | — | the P1 agent confirms each local layout. `cuda_kick_check.py:324` calls its plane 2 "trace opacity": if it feeds the engine as a trace plane, it is class N |

**Reads.** The fixtures above are writes. Raw-count READS against a local `Q`
(e.g. `assert ... <= Q`, totals divided by `Q`) are in the same two Q-class
files. The P1 agent converts them in the same pass, and the write lint's
sibling (V1) flags any `Q`/`65536` literal compared with a trace slice.

## 3. The factor, chosen by measurement

### 3.1 What real scenarios produce

Measured 2026-10-20 on this worktree's CPU build (`fire-12` `c8cf9da`) by the
scratchpad script `measure_trace.py`. Each tick it records every trace plane's
per-cell max and grid total, plus the bulk-N max. Fires are lit with
`debug_ignite`'s own law on every flammable tile.

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

What the measurements say:
- **Real play** stays below 0.3 units per cell and 10 units per grid. Fire
  makes very little smoke at today's yield. The combustion fraction is capped
  at 1.0, so if Erik raises it, at most twice today's.
- **The abuse cases are legal**, and RL agents will find them.
- **The miasma equilibrium** is `rate / decay ≈ 2100` units in the room, and
  about 230 units per cell in a 3×3 closet.
- **Abuse composes with compression.** Stage 3b preserves the trace-per-air
  ratio, so a blast's 18× bulk compression compresses the trace with it.
  Grenade spam under that compression gives about 197 units (A10).
- **So the saturation WILL bind in reachable abuse at k = 8**, and at any
  factor above about 4 bits. It is a primary mechanism, gated as one (V6),
  not a corner case. The factor only decides how far beyond real play it sits.

### 3.2 What the visible range needs

The decay law on one smoke cell (scratchpad `decay_law.py`, re-derived
independently by lens 1): an integer cell started at τ = 4 per tile, against
the configured e-fold. The table gives the time to cross each τ band, in
seconds. "10 % band" is the τ above which every value's per-tick loss is within
10 % of the exact exponential.

| k | rounding | 4→1 | 1→0.1 | 0.1→0.03 | 0.03→0.01 | 0.01→0 | 10 % band from τ |
|---|---|---|---|---|---|---|---|
| — | **configured exponential** | 173 | 288 | 150 | 137 | ∞ | — |
| 0 | ceil (today) | 61 | 18 | 1 | 0 | 0 | 55 |
| 8 | ceil | 171 | 273 | 123 | 88 | 52 | 0.22 |
| 8 | **nearest, floor 1 (shipped by P3)** | 172 | 286 | 155 | 104 | 52 | 0.11 |
| 9 | nearest, floor 1 | 172 | 286 | 149 | 145 | 104 | 0.054 |
| 10 | nearest, floor 1 | 172 | 286 | 150 | 137 | 197 | 0.027 |
| 12 | nearest, floor 1 | 172 | 286 | 149 | 136 | 370 | 0.0068 |

Today's law takes 80 s from black to nothing, against about 750 s configured
before the last trace.

### 3.3 Headroom per factor

The per-cell int32 ceiling is `2^31 / 2^(16+k)` units.

| k | ceiling (units) | × real max (0.29) | × grenade spam (11) | × miasma equilibrium (39) |
|---|---|---|---|---|
| 8 | 128 | 440 | 11.7 | 3.3 |
| 9 | 64 | 220 | 5.8 | 1.6 |
| 10 | 32 | 110 | 2.9 | **0.8** |
| 12 | 8 | 28 | **0.7** | **0.2** |

### 3.4 Decision: `TRACE_FINE_BITS = 8`

**Why a power of two.** The unit's meaning never changes. Every integer
reader's result on `S_old·2^k` is bit-identical (§2.4). And `k = 0` IS
today's engine (§6.2).

**Why k = 8:**
- **Headroom.** k = 8 keeps every single measured case below the saturation
  with at least a 3.3× margin. k = 9 halves that margin. k ≥ 10 saturates in a
  single measured legal case. No k keeps composed abuse below it (§3.1).
- **Decay.** With nearest rounding, k = 8 follows the configured decay within
  3 % from black down to τ = 0.03 per tile (3 % opacity per tile), and within
  25 % to τ = 0.01 (1 %). The last 1 % haze then leaves in 52 s instead of
  never. Nothing a player can see goes opaque-then-gone.
- **What k = 9 would buy.** Only the 0.01–0.03 band (1–3 % opacity per tile,
  the edge of visibility), at half the headroom. The headroom is worth more in
  an RL environment.

**The issue's arithmetic, redone at k = 8:**

| τ per tile | counts | exact loss per tick | removed by nearest rounding, floor 1 |
|---|---|---|---|
| 0.01 (just visible) | 1250 | 0.42 | 1: the edge of the linear regime |
| 0.03 | 3750 | 1.26 | 1 (averages out over the band) |
| 0.1 | 12 500 | 4.2 | 4 |

- Diffusion acts from |Δτ| ≥ 0.0019 per tile, against 0.49 today.
- Poison at its 0.05 dose threshold holds 838 861 counts and loses 141 counts
  a tick: fully exponential.

## 4. Where the scale lives

**One constant, one owner per language, twins held equal by a test.**

**The Python twin is a literal, not #78's read-through.** `optics_fixed.py`
reads `fine_bits` from the bound engine and returns 0 without one. That would
be a trap here: GameMaps without an engine (level load, most unit tests, the
tools) would silently read the trace planes at k = 0. `gas_fixed` is
import-light and serves engine-less GameMaps, so it carries its own literal,
held equal to the engine's by V1 (C5).

**C++** (`cpp/src/fixed_point.h`, namespace `fixedpoint`, FP_HD, the one kit
both backends compile):

```
constexpr int TRACE_FINE_BITS = 8;                          // #88: the trace planes' currency
constexpr int TRACE_SHIFT     = FP_SHIFT + TRACE_FINE_BITS; // 24
constexpr int32_t TRACE_CELL_MAX = INT32_MAX;               // the counted saturation (§5.3)
FP_HD int64_t trace_from_bulk(int64_t q);   // q * (1 << TRACE_FINE_BITS), assert(q >= 0): a bulk-count mass entering a trace plane (the soot mint)
FP_HD int64_t trace_density_shr(int64_t s); // s >> TRACE_SHIFT, s >= 0: Σ coef_q16 · S -> a Q16 fraction (sweep finishes, laser, readout)
FP_HD int32_t trace_add_sat(int32_t s, int64_t add, int64_t* clip);  // §5.3
int64_t quantize_trace(double v);           // round-half-away at 2^TRACE_SHIFT, returns INT64, never narrows (A3); host only
```

Every C++ site in §2 calls these. No `<< 8`, `>> 24` or `>> FP_SHIFT` on a
trace value is written anywhere else. In particular, the two
`radiation_sweep.h` finishes contain no `FP_SHIFT`, and every C++ trace
reader reaches the shift through them (C13).

`quantize_trace` is the one float in the kit, in `quantize`'s door-2 idiom:
- the double → int64 conversion is range-checked, since an out-of-range
  double → int is UB in C++;
- a value at or beyond int64 saturates to the int64 limit, after which the
  int32 saturation books it.

Bound by the binding as `breach_physics.TRACE_FINE_BITS`.

**Python** (`src/simulation/gas_fixed.py`, the Q16 boundary module for gas),
three currencies, three name families:
- **trace**: `TRACE_FINE_BITS = 8`, `TRACE_SHIFT`, `TRACE_ONE = 1 << 24`,
  `TRACE_CELL_MAX`, `quantize_trace` (int64 array), `quantize_trace_scalar`
  (Python int), `dequantize_trace`, `dequantize_trace_f32`,
  `trace_density_shr`, `trace_to_q16`, `trace_from_bulk`, `trace_add_sat`;
- **bulk**: `BULK_ONE`, `quantize_bulk`, `quantize_bulk_scalar`,
  `dequantize_bulk`, `dequantize_bulk_f32`;
- **Q16 fractions and temperatures** (A12): `Q16_ONE`, `quantize_q16_scalar`,
  `dequantize_q16`. Filter efficiencies are validated against `Q16_ONE` (V1);
- **deleted**: `FP_ONE`, `FP_ONE_F`, `SMOKE_MAX_Q`, `quantize`,
  `quantize_scalar`, `dequantize`, `dequantize_f32`. Every caller moves to the
  name of its own currency, and a missed one raises `AttributeError`.

**Reference** (`sweep_ref_q.py`): `TRACE_FINE_BITS` and `trace_density_shr`,
guarded equal to the engine's (§7).

**Digest spec** (`tests/field_digest_spec.toml`): `trace_fine_bits` (§6.1).

**Coefficients stay Q16 per REAL unit** (`heat_absorb_q16`,
`light_absorb_q16`, `light_glow_q16`, `beam_absorb_q16`). The density laws
absorb the extra bits in their one shift, `trace_density_shr`. That way:
1. the coefficients keep their full precision (a Q16 `light_glow` of
   0.22 / 256 would be 56 counts, a 0.57 % rounding, A11);
2. the door's `HEAT_ABSORB_MAX` headroom argument is untouched (each term
   `< 2^28 · 2^31 = 2^59`, 16 terms `< 2^63`; S is still int32);
3. `gas_medium.py`'s `/65536` on a coefficient stays correct.

## 5. Conservation and books

### 5.1 The identity, closed with no residual

Design v2 §3's identity was a bracket: its "deposits" term had no book
(§16 B1). After #88, per tick and per plane, exactly in int64, in the plane's
own currency:

    Σ S(after) − Σ S(before) ==  mint + deposit
                                − vent − wipe − sink − decay − clip
                                − [pump extraction + FieldEdit REMOVE + explosion clear]

The bracketed term is ACCEPTED GAP G6: those three destroy trace with no book
(today, and after #88). An identity test therefore runs in scenarios without
them, and then closes with **no residual**.

**The two new books** (orchestrator's B1 ruling), both written at the
ATTEMPTED amount, before the saturation:

| book | home | written by | sign |
|---|---|---|---|
| `trace_mint_sum[g]` | engine side, beside the trace books (`EOSSolver`, CUDA counter block, slot APPENDED) | the soot mint (CPU + CUDA) and the W5 puff | ≥ 0 |
| `trace_deposit_books[g]` | `GameMap` (Python ints) | the FieldEdit combine: the positive change an ADD or MAX edit attempts, `quantize(old + contribution) − old` when positive; any other Python-side insertion | ≥ 0 |

**The window (B7).** Every trace book (vent, wipe, sink, decay, clip, mint,
deposit) is **per tick, reset at the head of `Simulation.step()`'s physics
slot**: the window `sim.step()` brackets, which the v2 identity tests already
use.
- The engine books are reset at `EOSSolver::step` entry
  (`eos_solver.cpp:351-357`). The soot mint (combustion, after `run_substeps`,
  `physics_runner.py:939, 976`) and W5 fall inside the same tick's window. That
  order is now documented on the books and pinned by V6.
- The Python books are reset by `Simulation.step()` at the same point.
- CUDA slots are APPENDED to the counter block, never inserted (the P5c
  slot-12/13/14 precedent).

### 5.2 Why every existing book stays exact

| channel | law | exact after #88 because |
|---|---|---|
| advection (3b) | the debit and credit of a face are the SAME `price_face` int64 | unchanged law; S still int32 |
| vent | the donor books what it prices or diffuses onto vacuum/ring | the same int64, now in trace counts |
| wipe (3c), sink | the whole S of the cell | unchanged |
| diffusion | `F(i,j) = −F(j,i)`, magnitude-truncated, `c·|ΔS| < 2^45` | unchanged |
| decay | `Σ lost`, `v·frac < 2^47` | unchanged in form; P3's rounding changes the amount, not the booking |
| mint, deposit, clip | §5.1, §5.3 | new; attempted − landed == clip, per site |
| books' range | per tick; worst case a full map at `INT32_MAX` | `2^15` cells × `2^31` = `2^46` per tick, far below `2^63` |
| bulk and energy books | the soot mint keeps the bulk-count `soot`; P0 keeps the bulk split's `want` | bit-identical (V4, V5) |
| CUDA books | the `unsigned long long` `atomicAdd` of signed values is two's-complement modular | order-free (§15) |

**The tail's debug-only assert `Σ_map S < 2^31`** (`bulk_transport.cpp:802-807`)
is **deleted**, not re-derived. At k = 8 it is false in measured legal play
(the miasma room holds 2062 units). The counted saturation replaces it as the
argument for the narrow.

### 5.3 The counted int32 saturation, `TRACE_CELL_MAX = INT32_MAX`

**Why it exists.**
- Today's narrows `assert` in Debug and wrap in Release. The CUDA one does not
  even assert (`cuda_bulk_transport.cu:473`). The Python guards raise
  `OverflowError` mid-tick, so `seal_tiles` would refuse to close a door on a
  smoky tile.
- At k = 8 a cell reaches `2^31` in reachable abuse (§3.1).
- A wrapped trace cell goes negative: the sweep reads it as clear, and the
  poison dose as nothing. For an RL environment that is a found exploit.

**What it is.** One kit function per language, `trace_add_sat(s, add,
&clip)`:
- Compute `t = s + add` in int64.
- **It saturates from above only.** If `t > TRACE_CELL_MAX`, it stores
  `TRACE_CELL_MAX` and adds `t − TRACE_CELL_MAX` to `clip`.
- A negative `t` is a **contract violation, never clamped** (A4: clamping
  would mint unbooked mass). The CPU keeps its `assert`, and Python raises.
  CUDA counts it in an appended diagnostic slot `trace_neg_hits`, which every
  CUDA gate asserts is 0.
- At stage 3b it is applied ONCE per cell, on the net `s_own + ds`, never per
  face.
- It is pure integer and FP_HD, identical on CPU and CUDA. The soot mint and
  3b each write only their own cell, so it is order-free on the device (§15).

**Where it sits.** Every site that can add to a trace cell:

| site | clip book |
|---|---|
| FieldEdit trace ADD/MAX (`field_edit._combine_gas`) | `GameMap.trace_clip_books[g]` |
| stage 3b narrow (`bulk_transport.cpp:540-543`, `cuda_bulk_transport.cu:473`) | `EOSSolver.trace_clip_sum` (CUDA: appended slot) |
| soot mint (CPU + CUDA), W5 puff, the W3 receiver add | `trace_clip_sum` (through the pointer PhysicsEngine wires, as it wires `boundary_flux_`) |
| `seal_tiles` and `_inject_gas_n_core` (the vents' supply inject) on trace slices, inside the raise-free mutation pass | `GameMap.trace_clip_books[g]` |

Diffusion is convex and decay only removes, so they need no saturation.
`unseal_tiles` cannot exceed an existing cell (§2.3).

**What it is, against D1** (B3's wording). It is D1's option (b) at the int32
representability limit, 128 units, 440× the real maximum. It never cuts smoke
already on the tile, which was the substance of Erik's (a): `min(s + add, MAX) ≥
s` for `s ≤ MAX`. It caps only an addition that cannot be represented, and it
books it. The orchestrator has put it to Erik as a vetoable statement; the #88
closing comment says so in one line (B9).

## 6. Determinism and goldens

### 6.1 What moves, and the spec

**What moves at the flip:**
- the field digest's `gas` bytes, in every scenario with trace;
- **`GOLDEN_AGGREGATE`**. Its canonical scenario seeds smoke 0.6
  (`field_ab_harness.py:136`), and 0.6 quantizes at 2^24 to
  `10066330 ≠ 39322·256`. Through `a_gas`, `temperature` and `gas_energy` can
  move too. The decay rounding moves it as well: at the 0.6 seed, ceil
  removes 14 counts and nearest 13 (§16 B4);
- every CUDA check that reads `GOLDEN_AGGREGATE`;
- the vent ENTITY_SECT rows.

No float field and no unit state moves unless the smoke reaches it.

**Spec version → 7.**
- Dtype, shape, membership and order are unchanged, but a v6 and a v7 digest of
  the same world are not comparable. That is what the version exists to refuse.
- So v7 adds `trace_fine_bits = 8`, and `field_digest.py` asserts it equals
  `gas_fixed.TRACE_FINE_BITS` at import.
- The spec's CHANGE PROCEDURE comment (`field_digest_spec.toml:6-9`) is amended
  to "…if a field's dtype/shape/membership/order, or its integer CURRENCY
  (`trace_fine_bits`), changes" (C6).
- The serialization is unchanged.

### 6.2 ONE golden event, with the three-stage sole-cause proof

There is one re-baseline for the whole branch, in the gate-h lineage of
`tests/_xarch_perfield_digest.py`. It lands in **P3**, which takes the flip
(k 0→8), the decay rounding (ceil→nearest) and the CUDA re-greening together as
one gated patch: one writer, both builds, one lineage entry.

**This is my choice, made against two alternatives:**
- **Merging three patches** removes the known-red window C11 flags. That window
  would be CUDA red after a CPU-only re-baseline, plus an intermediate golden
  that never ships.
- **Ordering the rounding change before the flip** would still move the golden
  twice, since rounding alone moves it at k = 0.
- So one patch it is.

**P0, P1 and P2 may not move the golden.** P0 touches no system the canonical
scenario exercises. The scenario has no pump, vent or duct
(`field_ab_harness.py:120-151`). P1 and P2 run at k = 0 with ceil decay. **A
move in any of them is a bug** (B4).

**The proof, written into the lineage entry,** each stage reproduced twice:
1. **k = 0, ceil** (the P2 tree). It reproduces `f6ba810e…` on CPU and CUDA.
2. **k = 8, ceil.** It gives an intermediate value **X**, recorded in the entry
   with the fields that moved and their first tick, and **not committed as the
   golden**.
3. **k = 8, nearest.** It gives **Y**, the committed golden, again with the
   fields that moved relative to X.

The VERIFIED-SOLE-CAUSE paragraph names two causes, each with its own diff
(the constant; the one-line rounding). Both are reversible on the same tree.

**Reader invariance (V3).** It catches any reader that missed its
conversion, which k = 0 cannot see. P1's dry flip (§11) runs it before the
constant ever moves.

### 6.3 Where new is ALLOWED to differ from old × 2^8

| difference | why it is allowed |
|---|---|
| FieldEdit, W5 and the fixtures' seeds quantize at 2^24 | finer rounding of the same real amount: `|new − old·2^8| ≤ 2^7` counts per edit |
| **door thresholds** (`teargas_blind_density`, `poison_min_density`) quantize at 2^24 | each gate moves by less than one old count, toward the real-unit value (teargas: 0.4 old counts; poison by luck: 0) (A1) |
| the `price_face`, diffusion and W3 floors | they floor at the fine currency, so they lose up to 2^8 times less |
| decay | the purpose of #88 |
| diffusion acting where it was dead (|ΔS| < 240 old counts) | the same law, now resolved (D-B) |
| downstream of those: `a_gas` → temperature → `gas_energy`, light, poison and teargas timing | §6.4 sizes each one |

**Unchanged, restricted correctly (A9).** Bulk N and wind stay bit-identical
with or without trace, as do `gas_energy` and temperature when the sweep's gas
term is off (`heat_absorb = 0`). Since P5c, smoke-absorbed radiation lands in
`gas_energy`, so with the term on, the air sees the smoke through heat, by
design.

### 6.4 Intended behaviour changes (B5)

These are RL-visible, each with its size. All are intended; none needs a
HUMAN-TEST (Erik, 2026-10-20).

| change | size | cause |
|---|---|---|
| smoke thins gradually instead of opaque-then-gone | black → last trace: 80 s today, ~770 s after (§3.2); the configured e-fold within 3 % down to τ = 0.03 per tile | decay resolved, nearest rounding |
| smoke spreads in still air at visible densities | d = 0.10 tile²/s: about 4.5 tiles over 100 s, where today it barely moves below Δτ 0.49 | diffusion resolved (D-B) |
| **poison's lethal window at the dose edge** | about **1.7×** longer: from 0.06 down to the 0.05 threshold takes 27 s today (linear, 1 count a tick) and 46 s after (`ln(1.2)/0.004`) | decay resolved |
| poison and teargas diffusion | **no footprint change**: their counts were already above the 240-count diffusion edge (0.0037 units), so the newly-live band sits below every dose threshold | — |
| **teargas BLINDED near its threshold** | about 20 % longer: at 0.15 today removes `ceil(4.05) = 5` counts a tick, against 4.05 exact | decay rounding |
| the teargas gate's threshold | 0.4 old counts lower in real units (closer to 0.15 exactly) | A1 |
| **pumps and vents in smoke (P0)** | in the densest real smoke on ambient air (0.29 units over 1.0), **+29 %** air pumped and +29 % smoke removed. An airlock's `at_target` latches sooner in smoky rooms; compressed tiles see less change | trace leaves the bulk budget |
| radiation through smoke (`a_gas`) | rounding only: `a_gas` is bit-identical on equal smoke; it differs through the different smoke field | follows the rows above |
| the counted saturation | binds only in abuse (≥ 128 units in a cell) | §5.3 |

## 7. The integer reference

The sweep's gas term reads the trace planes, so per the reference's rule
`sweep_ref_q.py` changes FIRST, in P1, before the C++.

**What changes:**
- `TRACE_FINE_BITS` and `trace_density_shr` are added.
- `gas_extinction_q` and `gas_extinction_exp_q` shift by `16 +
  TRACE_FINE_BITS`. The glow coefficient follows.
- `gas_density_sum`'s term bound is unchanged.

**The gates must stay non-vacuous and must see fine counts** (A8).
`sweep_ref_q_gates._gas_planes` (`:104-106`) and the harness's `random_gas`
(`tests/_radiation_sweep_harness.py:114-132`, C1) draw
`trace_from_bulk(v) + rng.randrange(256)` on non-zero **trace** cells (smoke,
steam), with the bulk pair and `n_bulk` untouched. So gate 0 sees low bits that
are not multiples of 256, and can tell `>> 24` from a rounding variant.

**How the gates run:**
- Every gate prints its count of absorbing cells, which must be > 0.
- They re-run at k = 0 in P1, with identical results, and again in P1's dry
  flip and P3 at k = 8 (all PASS, non-vacuous).
- `tests/test_radiation_sweep_reference.py` (gate 0) and
  `tests/test_temperature_gas_radiation.py` follow.
- A guard asserts the reference's `TRACE_FINE_BITS` equals the engine's.

## 8. CUDA

**What changes on the device:**
- **Stage 3b/3c.** One kernel (`bulk_flux_energy_transport_device`) serves
  both chained and resident.
- **The resident tail** (`trace_tail_sink` / `_diffuse` / `_decay`) needs only
  P3's rounding, through `trace_decay.h`.
- **The soot mint and the sweep's gas terms** call the same FP_HD kit as the
  CPU.
- **The new books.** `trace_clip_sum`, `trace_mint_sum` and `trace_neg_hits`
  are APPENDED slots in the trace counter block (`cuda_eos_resident.cu:930-943`,
  `cuda_eos_step.cu:144-155`), D2H'd with the others.

**What V9 cannot see** (C12). A mis-scaling inside a shared FP_HD kit function
is identical on both backends by construction, so GPU == CPU passes. Only V3
sees it.

**Gates at tolerance 0** through the CUDA harness (`run_cuda_script`, never
importing the CUDA .pyd into pytest):
- `cuda_trace_smoke_check.py`: chained and resident, every plane and every
  counter including clip, mint and neg_hits;
- `cuda_combustion_check.py`, `cuda_radiation_sweep_check.py` (light included)
  and `cuda_s8a_check.py`;
- the thermal-mass-eos resident leg;
- the saturation cases of V6 on CUDA (3b chained, 3b resident, mint).

Every `GOLDEN_AGGREGATE` CUDA consumer is green at P1 and P2 with an unchanged
value, and at P3 with Y.

**The V1 twin guard also runs inside `run_cuda_script` against the CUDA
module**, so a stale `build_cuda` fails V1 at the flip (C4).

## 9. Look-side consequences (render only: listed, not tuned)

The renderer reads dequantized floats, so its inputs keep their units. What
changes is the field it is shown (§6.4). The smoke-tuning skill's §6 list
stands, retuned once #88 is in:
- `cov_tau_*`, `cov_soft_*`, `haze_*`, `height_tau`, `top_light_*` (possibly
  to 0);
- the boil range, rechecked after #12's fire tuning;
- **added by #88:**
  - the speckle and `gas_relief` clips, which saw only coarse 2^-16 steps at
    thin densities;
  - `effect_gas_floor`, the poison/teargas legibility floor, since poison now
    lingers at its configured rate near the threshold;
  - **the smoke shadow and glow-through** (#93 G2), which were tuned on a
    field that went opaque and then vanished.

Doc for the list: `docs/graphics_direction_2026-10-18.md` §11. P6 adds a dated
line to the skill.

## 10. Verification plan (properties)

Each item names the property and the change that must break it. Nothing pins a
count, a set of gases, or a snapshot value.

- **V1, the currency is closed.**
  - **The twins agree.** `breach_physics.TRACE_FINE_BITS` (the CPU module and,
    inside `run_cuda_script`, the CUDA module), `gas_fixed.TRACE_FINE_BITS`,
    `sweep_ref_q.TRACE_FINE_BITS` and the spec's `trace_fine_bits` are all equal.
  - **Lints over `src/`, `renderer/`, `tools/` and `tests/`:**
    - the deleted `gas_fixed` names are referenced nowhere;
    - **the write lint:** any assignment or `+=` into a trace slice
      (`gas[<trace id>]`, `.smoke`, `gas[:N_TRACE_GASES]`) whose right-hand
      side takes no `trace_*` name and is not a `# raw trace counts:
      scale-free (#88)` fixture fails;
    - no `bulk_*` / `quantize_bulk*` / `dequantize_bulk*` call takes a trace
      slice;
    - no sum over a trace slice runs without an int64 accumulator (A2);
    - the filter validator bounds against `Q16_ONE`;
    - the two `radiation_sweep.h` finishes contain no `FP_SHIFT` and call
      `trace_density_shr` (C13).
  - *Breaks if* a twin moves alone, an ambiguous name returns, a trace slice is
    written or read in another currency, or a CUDA module is stale.
- **V2, k = 0 is today.** A report, not a standing test: the P1 and P2 trees
  reproduce `f6ba810e…` on CPU and CUDA, and stage 1 of §6.2's proof restates
  it.
- **V3, reader invariance, through the public entry points** (C2). The test
  holds a **reader registry**:
  - **the registered readers:** `pack_hover_readout`; the sensor accessor's
    `sample` and `area`; `exchange`'s teargas and poison rows; `fire_beam`;
    `PhysicsRecorder.record`; the engine's sweep (one tick, reading `a_eff`,
    light `a_c` and glow `g_c` through the binding); and the renderer's
    `_trace_floats`.
  - **the subset lint:** every `src/` / `renderer/` module that reads a trace
    slice (V1's regex) appears in the registry. This is a classification check,
    not a count, so the registry grows freely.
  - **the seeding:** each reader runs on a real GameMap whose trace planes are
    seeded with `S_old << k`, over a sweep of `S_old` that covers each
    threshold, the ONE cap, and a negative cell.
  - **what it asserts:**
    - integer readers are bit-identical to the pre-#88 integer law on `S_old`,
      written into the test as the documented expression;
    - float readers are float-identical;
    - the two **gates** are asserted by their real-unit law instead (A1):
      `gate(S) ⇔ S ≥ quantize_trace_scalar(threshold)`, monotone in S.
  - *Breaks if* any call site skips, doubles or mis-names its conversion, or a
    new trace reader is unregistered.
- **V4, writer scale.**
  - The soot mint into an empty cell writes exactly `trace_from_bulk(soot)`,
    with the bulk N2 and `e_shed` bit-identical to the shift-free computation.
  - A FieldEdit ADD of amount `a` onto a tile holding `v ≥ 0` changes it by
    `quantize_trace_scalar(a·w)`, within one count (A11: float64 ties near .5
    may differ by one).
  - *Breaks if* a writer mints in the bulk currency, or the soot split moves the
    bulk books.
- **V5, no mixed currency (P0).**
  - An extraction from a smoky tile removes the same bulk N as from a clear tile
    with equal bulk, and the bulk-energy book `pump_extract` equals
    `want·T_abs` exactly in both (B8).
  - The trace taken satisfies `take_t·N ≤ bulk_taken·S_t < (take_t + 1)·N`.
  - A tile with `N_bulk == 0` and `S > 0` takes no trace and raises nothing (A5).
  - For a vent, the plenum's `trace_raw + sink` grows by exactly the trace
    taken (B8).
  - *Breaks if* trace counts enter a bulk budget again, are priced against
    `N_total`, or the divide is unguarded.
- **V6, the books close, and the saturation holds at every site, per backend**
  (B1, C10, A3, A10).
  - **The identity.** On the v2 scenarios (sealed blast, seal-then-breach,
    vent supply) **plus a soot-minting fire and a W5 boil**, the §5.1 identity
    without G6's terms is exact per tick per plane, with no residual.
    `test_water_boil.py` reads its puff from `trace_mint_sum`, not from the
    bracket.
  - **The saturation, parametrized over §5.3's sites.**
    - CPU: FieldEdit, 3b, mint, W5, the W3 receiver, `seal_tiles`,
      `_inject_gas_n_core`.
    - CUDA: 3b chained, 3b resident, mint.
    - Plus one deliberate overflow at a conversion: a 200-unit W5 puff, and a
      200-unit FieldEdit, into an empty cell.
  - **Each case asserts:** the plane is ≤ `TRACE_CELL_MAX` and ≥ 0; `clip`
    equals the exact excess; `trace_neg_hits` is 0; and the identity closes.
  - **The composite abuse** (grenade spam under a blast's compression) is a
    primary case, not a corner.
  - *Breaks if* any single site wraps, raises, narrows before the saturation,
    clamps a negative, or drops without booking.
- **V7, the configured decay where visible (P3).**
  - A smoke cell from τ = 1 to τ = 0.03 per tile reaches each τ within 5 % of
    `ln(τ0/τ)/decay`.
  - Poison at `poison_min_density` decays at its configured rate within 5 %.
  - A gas with `decay = 0` loses nothing: `lost(v, 0) == 0` inside the kit
    (B6).
  - Every cell reaches 0 in bounded time. Design v2 V8's test is re-bounded from
    the law (exponential to the linear edge, plus that many ticks), not from
    `range(v + 1)` (B6).
  - *Breaks if* the resolution is lost, the rounding reverts to the linear
    regime, or the floor of 1 decays a non-decaying gas.
- **V8, diffusion resolves visible gradients.** In still air, a blob at
  τ = 0.3 per tile has a growing second moment within 10 s, conserves `Σ S`
  exactly, and keeps a uniform field uniform. *Breaks if* the currency is
  coarsened back.
- **V9, CPU == CUDA at tolerance 0**, chained and resident, on every trace plane
  and every counter including clip, mint and neg_hits, **with zero skips under
  `pytest -rs`** (C4). *Breaks if* a backend diverges or a gate goes vacuous.
- **V10, the door.** A config giving a conservative row a non-zero
  `heat_absorb`, `absorption` or `scatter_albedo` is refused, with a message
  naming #88. *Breaks if* a bulk plane could enter a trace-currency density
  sum.
- **Carried over unchanged:** design v2's V1–V9, their fixtures moved per
  §2.7. V5's "air does not see the smoke" is restricted per §6.3 (A9).

**What each gate catches and misses** (C12, folded):

| gate | catches | its known blind spot, and what covers it |
|---|---|---|
| V5 | trace in the bulk budget; pricing against `N_total` | the taken trace landing in no book: G6, recorded |
| V2 (k = 0) | any semantic change at k = 0 | missed conversions: P1's dry-flip V3 |
| V1 | twin drift, deleted names, wrong-currency calls, stale CUDA module | — |
| V6 | wrap, raise, early narrow or unbooked drop at each site | — |
| V3 | a skipped, doubled or mis-named conversion at a registered reader | an unregistered reader: the subset lint |
| V9 | GPU ≠ CPU | a shared FP_HD mis-scaling: V3 |
| V7 | the linear regime returning; decay of a non-decaying gas | — |

## 11. Patch plan

**Rules for every patch:**
- **Where it runs.** In
  `C:\Users\steen\projects\breach\.claude\worktrees\88-smoke-unit-rescale`
  itself, **with no `isolation`** (an isolated agent worktree is cut off MAIN
  and holds no builds), one writer at a time.
- **Tests.** Python is `C:/Users/steen/anaconda3/python.exe`; tests are
  `pytest tests -q`.
- **HUMAN-TEST.** None (Erik, 2026-10-20).
- **Critics** ran one at a time before P0 (§14).

**The C++ build discipline. MUST, in every patch that touches C++ (P1, P2,
P3)** (C4; incident: a CUDA gate SKIPS green when `cpp/build_cuda` is
missing, and passes on a stale module):
1. Run `cpp\build_cpu_home.bat` (`--fresh`, a full rebuild) and
   `cpp\build_cuda.bat` (Ninja, CUDA 12.4), and record both `BUILD_EXIT=0`
   lines in the report.
2. Check that both `.pyd` files are newer than the newest `cpp/src` file, and
   report the timestamps.
3. Run `pytest tests -q -rs`. The report shows **zero skips** among the CUDA
   gates (`-k cuda`). A skip is a gate failure.

| # | content | gate (oracle) | tier |
|---|---|---|---|
| **P0** | Trace leaves the bulk split (§2.2): `extract_gas_n_vec` splits over the bulk planes only, prices trace by air with the `N_EPS` guard, and keeps today's destroy-on-extract (G6); `extract_gas_n` returns the bulk total; the `inject_gas_n_vec` gate stays. The vent and pump tests are re-checked as properties (§2.7 class R). Python only | V5; the vent and pump suites; **`GOLDEN_AGGREGATE` reproduced: a move is a bug** | **Opus** (the vent tests may pin values: rewrites need judgement) |
| **P1** | The currency **at k = 0**. The C++ kit and binding (§4); the `gas_fixed` split into trace / bulk / Q16 names with the deleted names; every caller moved (§2, the bulk and Q16 users included); the reference FIRST (§7), with its gates re-run and A8's draws; every reader conversion (§2.4) including `_trace_floats` and the sensor's `_channel_array`; the door (V10); the fixture inventory moved (§2.7) and `test_ps1_smoke_roundtrip` rewritten; V1's guard and lints; the V3 registry test. **The DRY FLIP** (C3): after the golden reproduces, build the same tree at `TRACE_FINE_BITS = 8` locally, uncommitted, on CPU and CUDA, and run V1, V3, V4 and every reference / `random_gas` gate's absorbing-cell count. A failure is a P1 bug, fixed in P1, and the dry flip is re-run. Then revert to 0 | **V2: golden unchanged on CPU and every CUDA check**; full suite green at k = 0; V1, V10; the dry flip's V3/V4/V1 green at k = 8; build discipline | **Opus** (wide; the call-site choices are judgement, and the dry flip is the mutation catcher) |
| **P2** | The counted saturation at k = 0: `trace_add_sat` at every §5.3 site on CPU and CUDA, applied once per cell on the net and from above only (A4); the books `trace_clip_*`, `trace_mint_sum`, `trace_deposit_books`, `trace_neg_hits` with their per-tick window (§5.1), CUDA slots appended, bindings; the `Σ_map` assert deleted; the Python OverflowErrors on trace slices replaced inside the mutation pass | V6 (every site, both backends, the composite abuse); **golden unchanged**; build discipline | **Opus** (the placement and window calls are judgement, and the oracles see only the planted cases, C11) |
| **P3** | **THE FLIP, one gated unit** (§6.2): `TRACE_FINE_BITS` 0 → 8 in the four twins; `trace_decay::lost` → `min(v, max(1, (v·frac_q + 2^15) >> 16))` with `frac_q <= 0 → 0` inside the kit (B6), and the header plus design v2 §2.3 / D4 cross-referenced; spec v7 and its procedure comment (C6); CUDA re-greened, chained and resident; **the ONE `GOLDEN_AGGREGATE` re-baseline** with the three-stage proof (X recorded, Y committed, each reproduced twice) and the fields-moved paragraphs; V7, V8; design v2's V1–V9 at k = 8 | the full CPU suite and every CUDA gate at k = 8; V7, V8, V9; the lineage entry's two sole-cause diffs; build discipline | **Sonnet** (after P1's dry flip the diff is the constant, one line of rounding, and the spec: mechanical under strong oracles. Any test that goes red at the flip is a behaviour change to report against §6.4, never bent) |
| **P4** | Close: the CLAUDE.md rows (§12 b), including the corrected "already-booked" sentence (G6) and the Gas pump row; the survey row; the smoke-tuning skill's §6 dated line (§9); the #88 comment (the saturation as one line for Erik, B9); Roadmap #46; `git merge --no-ff` into fire-12 on green | suite green | inline (orchestrator) |

## 12. Systems

**(a) Existing canonical systems this design uses:**
- **Trace transport**: the law unchanged, the narrow saturated and counted, the
  decay rounding amended.
- **The fixed-point kit**: the trace currency lives there.
- **The Q16 boundary modules**: `gas_fixed.py` splits into trace, bulk and Q16
  names.
- **FieldEdit**: the one trace deposit door, now booked.
- **Gas pump primitives** and the **vent / duct system**: P0 removes the mixed
  split.
- **Material / gas tables**: the door's bulk-coefficient refusal.
- **The radiation sweep**, the **extinction planes** (the gas term), and the
  **integer reference**, which changes first.
- **The light accessor**: it reads unchanged floats.
- **The recorder**: contract unchanged.
- **The field digest / GOLDEN_AGGREGATE**: spec v7, one re-baseline.
- **The CUDA harness**, the **tile inspector**, the **sensor accessor**, the
  **dequantize convention** (the renderer's one `_trace_floats`).

No parallel system is created (§17 confirmed).

**(b) New systems, and the CLAUDE.md wording P4 writes:**

- **Trace currency** (a new row; `fixed_point.h` `TRACE_FINE_BITS` + kit, twin
  `gas_fixed.py` `trace_*`): "The five trace planes are int32 in the TRACE
  currency, `2^(16+TRACE_FINE_BITS)` counts per real unit (#88); the bulk
  planes in the same `gas` array stay Q16, and Q16 fractions/temperatures have
  their own `q16_*` names. Every trace read or write converts ONCE through the
  kit (`trace_from_bulk`, `trace_density_shr`, `quantize_trace`,
  `trace_add_sat`) or `gas_fixed.trace_*`, never an inline shift and never a
  bulk name; sums over a trace plane accumulate in int64. Coefficients stay Q16
  per real unit. No sum mixes the two currencies: trace is priced by air, and
  the gases door refuses an optical coefficient on a bulk row. The twins and the
  digest spec's `trace_fine_bits` are held equal by a test (`gas_fixed` keeps a
  LITERAL twin — it serves engine-less GameMaps)."
- **Trace transport** (the row's clamp/books sentence is REPLACED; B3's
  wording, adapted to the B2 ruling): "No trace plane is clamped anywhere
  outside a reader, bar the counted int32 saturation (`TRACE_CELL_MAX`, #88) at
  every site that adds to a cell; trace ADD edits through FieldEdit are
  additive up to that saturation and never cut the tile (D1). Every loss is
  booked: vent in the trace slots of `boundary_flux_`, wipe, sink, decay,
  clip; every gain is booked: `trace_mint_sum` (soot, W5),
  `trace_deposit_books` (FieldEdit). Other movers (vent plenum ledger, W3,
  `destroy_wall`/`seal_tiles`) conserve or book through those. **Pump
  extraction, FieldEdit REMOVE and the explosion's smoke clear destroy trace
  UNBOOKED (ACCEPTED GAP, #88 G6)**." The old "separate, already-booked
  systems" clause is deleted: it was false for the pump and the clear.
- **Gas pump primitives** (a clause added, B3): "trace is priced by air
  (#88 P0): the split runs over the bulk planes; each trace plane leaves at
  the cell's `S/N_bulk`, and the pump destroys what it takes, unbooked (G6)."
- **Extinction planes** (amended): "the density sum is in the trace currency,
  finished by `trace_density_shr`; the door refuses a bulk row's optical
  coefficient".
- **Integer reference** (amended): names `TRACE_FINE_BITS`, and the gates draw
  fine counts.
- **Field digest** (amended): "…or its integer currency (`trace_fine_bits`)".

## 13. ACCEPTED GAPs and decisions

**ACCEPTED GAPs:**
- **G1, the decay fraction's own resolution.** `frac_q = quantize(decay·dt)` in
  Q16 is off by +0.7 % for smoke (22 vs 21.85), poison and steam, −1.1 % for
  teargas and −2.3 % for fuel_gas. A Q24 fraction would fix it, but that is a
  different change to the law. Recorded, not fixed.
- **G2, the faint tail is linear.** Below τ ≈ 0.01 per tile (1 % opacity),
  decay removes 1 count a tick, so the last haze leaves in about 52 s instead
  of never. This is D4's bounded-time property, kept.
- **G3, no bulk absorber.** The door refuses optical coefficients on the bulk
  rows. A future "burnt products absorb" row on inert_N2 (already a P5c ACCEPTED
  GAP) needs a second, bulk-currency partial sum in the sweep and the laser.
- **G4, the saturation binds in reachable abuse.** Two cases measured or
  composed: a Miasma Vent held in a closet (~230 units per cell), and grenade
  spam under a blast's compression (~197). It books the excess, and the dose is
  lethal there anyway.
- **G5, the soot mint is not finer.** It is minted in bulk counts and shifted,
  to keep the Dalton split with N2 and `e_shed` exact.
- **G6, unbooked trace destruction** (orchestrator's B2 ruling). Pump
  extraction (`pump_system.py:113` discards the trace `extract_gas_n_vec`
  takes), FieldEdit REMOVE and the explosion's smoke clear (`physics.py:69,
  98`) destroy trace with no book, today and after #88. P0 re-prices the pump's
  take, and keeps the destroy. Booking them is another system: the orchestrator
  files the issue. P4 corrects CLAUDE.md's false "already-booked" sentence, so
  the rule matches the code.

**Decisions:**
- **D-A (P0):** trace is priced by air on extraction. The sizes are in §6.4.
  It is required: without it, trace would eat 98.7 % of a pump's budget in
  smoke at k = 8.
- **D-B:** diffusion becoming live at visible densities is part of the rescale:
  the same law and dials, now resolved. Design v2 D6 assumed a diffusion it
  never measured working.
- **D-C:** the decay rounding is amended inside the flip patch (§6.2). It is the
  issue's own goal (Erik's 2026-10-05 ruling), and it needs one golden event,
  not two.
- **D-D:** P3 bundles the flip, the rounding and CUDA into one patch, so that
  exactly one golden event exists and no known-red window does (§6.2).

**Erik.** Nothing needs his decision under the delegation. The counted
saturation bounds his D1 ruling at 128 units, and never cuts the tile. The
orchestrator has told him as a vetoable statement, and the #88 comment repeats
it in one line (B9).

## 14. Critique rounds

| lens | verdict | blockers |
|---|---|---|
| arithmetic, overflow, determinism (Opus, 2026-10-20) | PASS WITH FIXES | none. 3 MAJOR to fix before P1/P3 (A1 the teargas gate is not reader-invariant, so V3 as written fails; A2 numpy 1.26 on Windows accumulates int32 sums in int32; A3 `quantize_trace` narrows to int32 before the rail can see the excess). 9 MINOR. See §15 |
| conservation, books, behavioural semantics (Opus, 2026-10-20) | PASS WITH FIXES | none. 4 MAJOR to fix before P0/P2 (B1 the §5.1 "deposits" term has no book, so a clip at a deposit site is invisible to the identity; B2 the pump's extracted trace and FieldEdit REMOVE destroy trace unbooked today, and P0 rewrites the pump lines without saying where its trace goes; B3 the rail IS D1's rejected option (b) at the representability limit, and §12's CLAUDE.md amendment leaves "additive with no ceiling (D1)" and "already-booked systems" standing; B4 two golden re-baselines break the branch's own gate-h precedent, and P0 cannot lawfully move the golden at all). 5 MINOR (B5-B9). See §16 |
| inventory, systems reuse, patch & verification plan (Opus, 2026-10-20) | PASS WITH FIXES | none. 4 MAJOR to fix before P1 (C1 the test fixtures have no inventory, and the silent ones bypass V1's lint: a local `Q = 65536`, `fire_fixed` on a trace plane, and the second reference vocabulary `random_gas`; C2 V3 tests reader functions, so a trace call site renamed to a `bulk_*` name passes V1, V2 and V3; C3 P1's oracle is blind by construction, and the plan defers the misses into the flip patch; C4 a CUDA gate SKIPS green when the worktree's `cpp/build_cuda` is missing or stale, and "worktree subagent" is ambiguous). 9 MINOR (C5-C13). See §17 |

### 14.1 Resolutions (v2)

All findings were accepted. B1–B4 follow the orchestrator's rulings of
2026-10-20, and the A- and C-findings were accepted as the critics proposed.

| finding | resolved in |
|---|---|
| A1 the thresholds are not reader-invariant | §2.4 teargas and poison rows; §6.3 "door thresholds"; §6.4; V3 asserts the gates' real-unit law |
| A2 numpy 1.26 accumulates in int32 | §2.1 int64 requirement; V1 lint; §12(b) rule |
| A3 `quantize_trace` narrows before the saturation | §4: int64 return, range-checked, `trace_from_bulk` multiply + assert; only `trace_add_sat` narrows; V6 200-unit cases |
| A4 the negative-result semantics | §5.3: from above only; negative = contract violation (assert / raise / `trace_neg_hits`); once per cell on the net |
| A5 P0's N_EPS guard | §2.2 extraction row; V5 |
| A6 keep the `inject_gas_n_vec` gate | §2.2 row: left as it is |
| A7 the sensor conversion point | §2.4 sensor row (`_channel_array`) |
| A8 the reference gates never see a fine count | §7: `trace_from_bulk(v) + randrange(256)` on trace planes only |
| A9 the `gas_energy` contradiction | §6.3 "Unchanged, restricted correctly" |
| A10 the headroom claim | §3.1 (composed abuse ~197 units), §3.4, G4; V6 primary case |
| A11 the arithmetic slips | §4 (0.57 %), §2.5 (`price_face` bound), V4 (within one count) |
| A12 the Q16 fraction names; `unseal_tiles` | §2.1, §4 `q16_*` names, V1 filter-validator lint; §2.3 `unseal_tiles` row |
| B1 the deposits had no book | §5.1: `trace_mint_sum` and `trace_deposit_books`, written at the attempted amount before the saturation; the identity closes with no residual; V6 |
| B2 pump / REMOVE / clear destroy trace unbooked | **ACCEPTED GAP G6** (orchestrator's ruling: not built in #88); §5.1 bracketed term; P0 keeps destroy-on-extract, re-priced only; P4 corrects CLAUDE.md's "already-booked" |
| B3 the rail is D1's option (b) | §5.3 wording replaced; renamed per C7; §12(b) Trace transport and Gas pump rows in the critic's wording, adapted to G6 |
| B4 two golden events; P0 cannot move the golden | §6.2: ONE event in P3, three-stage proof; P0/P1/P2 "a move is a bug"; D-C, D-D |
| B5 behaviour changes with sizes | §6.4 table |
| B6 the floor of 1 at `frac_q == 0`; the V8 loop bound | P3's row (`frac_q <= 0 → 0` in the kit); V7 |
| B7 the book window and sign | §5.1 window paragraph |
| B8 V5 pins the books | V5 (`pump_extract`, plenum `trace_raw + sink`) |
| B9 the line for Erik | §5.3 last paragraph; §13 "Erik"; P4 |
| C1 the fixture inventory; the silent classes; `random_gas`; `test_ps1_smoke_roundtrip` | §2.7 (inventory, classes, conversions); §2.2 test row; V1 write lint; §7 |
| C2 V3 tested functions, not call sites | V3 registry through public entry points, subset lint; V1 no-bulk-name lint |
| C3 P1's oracle was blind | P1's dry flip at k = 8 |
| C4 CUDA gates skip on a missing or stale build | §11 build discipline (MUST, with the incident); §8 twin guard on the CUDA module; V9 zero skips |
| C5 the #78 pattern misquoted | §4 "a literal, not #78's read-through" |
| C6 the spec procedure | §6.1; P3 |
| C7 "rail" collides | renamed throughout: the counted int32 saturation, `trace_add_sat`, `trace_clip_*` |
| C8 inventory misses | §2.1, §2.2 (hover `bulk_n`, `gas_energy` display), §2.4 tools row, §2.5 |
| C9 one renderer seam | §2.4 renderer row (`_trace_floats`) |
| C10 V6 per site and per backend | V6 parametrization |
| C11 order, tiers, splits | §11: P3 bundles flip + rounding + CUDA; P2 → Opus; P3 → Sonnet after C1/C3; P0, P1 Opus |
| C12 each gate's mutation | §10 table |
| C13 the V1 hand list | V1 lint on the finishes' property |

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

## 16. Critique — books & semantics lens

Scope: every trace book and every gameplay reader, read in the code at `b7425b8`.
The question for each: does it stay exact, and does any behaviour change that the doc
does not list?

**Confirmed sound (no finding):**
- **Engine books** (`eos_solver.cpp:348-357`). The four per-tick trace books (vent
  slots of `boundary_flux_`, wipe, sink, decay) are reset at `EOSSolver::step` entry
  and are per gas. No consumer sums `boundary_flux()` across gases: every reader in
  `tests/` indexes a gas or compares lists slot by slot (`test_air_boundary.py:826`,
  `cuda_trace_smoke_check.py:109`). So "each plane in its own currency" mixes nothing.
- **Every `N_EPS` gate is a bulk gate**: the 3b donor guard (`bulk_transport.cpp:483,
  488`), the 3c wipe (`:632`, CUDA `:360`) and the sweep's floor
  (`radiation_sweep.h:249, 275`). None compares a trace count to a threshold, so the
  rescale moves none of them, and P5's floor-1 decay meets none of them.
- **Vent plenum ledger.** `trace_raw` and `sink` are Python ints, serialized as
  `<q` (`entities/serialize.py:88-95`). ×2^8 is far from int64 limits. The filter
  scrub `(r·eff_q) >> 16` (`vent_system.py:400`) is a fraction of a count, so the
  sink book stays exact. Supply-side trace is `share·trace_raw // n_bulk`
  (`:478`), which is scale-free.
- **W3** (`physics_engine.cpp:1112-1135`) moves trace with the remainder to the last
  neighbour, so it conserves `Σ S` at any scale.
- **`seal_tiles`** (`gamemap.py:2766-2780`) uses divmod and conserves exactly. The
  rail goes inside the "raise-free by construction" mutation pass (`:2751-2755`),
  which fits that pin better than today's pre-check `OverflowError`.
- **`destroy_wall`**: trace is left to the tail's booked sink. Unchanged.
- **Soot mint**: the Dalton split and `e_shed` keep the bulk-count `soot`
  (`combustion.cpp:1003-1005`), so the energy and bulk books are bit-identical.
- **No vision reader.** No sim module outside the listed readers touches a trace
  plane. I grepped `src/simulation/*.py`: only `combat`, `exchange`, `field_edit`,
  `gamemap`, `gas_fixed`, `gases`, `physics`, `physics_runner`, `recorder`,
  `sensor_accessor`, `simulation` and `weapons` match, and all are in §2. Smoke does
  not block sight, so the rescale cannot change line of sight.
- **P0 is in scope and is required.** It is not smuggled.
  - Without P0, at k = 8 a tile holding real-max smoke (0.29 units = 4.87 M counts)
    over ambient bulk (65 536) splits a pump's budget 98.7 % trace, 1.3 % air
    (`gamemap.py:3097-3101`). An airlock pump in smoke would stall about 77×, and a
    duct would move smoke about 77× faster.
  - The all-slices split is D10's July rule (`docs/archive/arc_b_impl_2026-07-21.md`
    §6). D10 predates P-T0, which retired trace mass (`n_total == n_bulk`). P0 is
    P-T0's unfinished consequence on one primitive, and stage 3b and the vent supply
    side already follow it.
- **P5 leaves no residue.** `max(1, …)` capped at `v` reaches 0 in at most `v`
  ticks below the linear edge. Above it, nearest rounding is unbiased, where ceil is
  biased by +½ count a tick. That is the better reading of Erik's 2026-10-05 ruling.

### B1. MAJOR: the §5.1 "deposits" term has no book, so a clip at a deposit site is invisible to the identity

- **Evidence.**
  - FieldEdit's trace combine (`field_edit.py:499-510`) books nothing.
  - The soot mint `SOOT[s] += soot` (`combustion.cpp:1004`) has an energy counter
    (`e_soot_shed_sum`) but no mass counter.
  - The W5 puff `gas_slice[i] += puff_q` (`physics_engine.cpp:1045`) books nothing.
  - So today's identity is a *bracket*. `test_water_boil.py:74-90` measures the puff
    as `Δ Σ S + vent + wipe + sink + decay`, and v2's identity tests run only
    deposit-free scenarios (`test_smoke_transport_v2.py:91-92`: "No soot").
- **Consequence.**
  - Under a bracket, "deposits" means whatever the residual is.
  - A rail at a deposit site that clips without booking, books twice, or books into
    the wrong gas reads as "a smaller deposit", and the identity still closes.
  - These are exactly the sites where §5.3 says the rail binds (A10: grenade spam
    under compression; G4: the closet).
  - So V6's "`clip` books exactly the excess" has no closed identity to stand on.
    It can only be asserted per cell, against hand-computed expectations.
- **Fix (P2).** Book every mint at its attempted amount, before the rail:
  - `trace_mint_sum[g]`, per tick, for the soot mint and the W5 puff, on both
    backends. On CUDA, append it to the counter block beside `clip`.
  - `GameMap.trace_deposit_books[g]` for the FieldEdit net (ADD positive, REMOVE
    negative) and for any Python-side insertion.
  - §5.1 then becomes a closed identity with no residual:
    `Δ Σ S == mint + deposit − vent − wipe − sink − decay − pump − clip`.
  - V6 asserts that identity on the rail scenario. `test_water_boil.py` can then read
    the puff from its book instead of from the bracket.

### B2. MAJOR: the pump and FieldEdit REMOVE destroy trace with no book; P0 rewrites the pump lines and says nothing about where its trace goes

- **Evidence.**
  - `pump_system.py:113` calls `gmap.extract_gas_n(...)` and discards the result.
  - `extract_gas_n_vec` zeroes the trace slices it takes (`gamemap.py:3106-3108`).
    The only book it writes is the bulk energy book `pump_extract` (`:3109-3113`).
  - The explosion's smoke clear is a FieldEdit REMOVE-to-0 over the inner 40 % of the
    radius (`physics.py:69, 98`). It is also unbooked.
  - CLAUDE.md's Trace transport row says these movers are "separate,
    already-booked systems". For the pump and the clear, that is false today.
- **Why this lens cares now.**
  - P0 changes how much trace the pump takes: priced by air, up to (N+S)/N more.
  - The doc's own claim is that "every loss is booked" (§12 b).
  - At k = 8 these sites are also where a mis-scaled conversion would hide, because
    nothing counts them.
- **Fix (P0).**
  - The rewritten `extract_gas_n_vec` returns the per-slice vector as today. The pump
    books the trace part in `GameMap.trace_pump_books[g]` (a Python int, the
    `pump` term in B1's identity). The vents keep crediting their plenum.
  - FieldEdit REMOVE books its net into B1's deposit book.
  - If this is judged out of scope, record it as **ACCEPTED GAP G6** ("pump
    extraction and FieldEdit REMOVE destroy trace unbooked") and correct CLAUDE.md's
    "already-booked" sentence. Leaving the sentence as it is would be the stale-rule
    failure the master CLAUDE.md names.

### B3. MAJOR: the rail is D1's rejected option (b) at a different value, and §12's amendment leaves CLAUDE.md contradicting the code

- **Evidence.**
  - D1 as put to Erik (`smoke_transport_design_2026-10-04.md` §11): "(b) Keep the
    ceiling, counted as destruction". Erik ruled (a): "a deposit adds exactly its
    amount, with no upper clamp".
  - §5.3's `trace_add_rail` at FieldEdit is (b) with the ceiling at 128 units
    instead of 1.
  - §5.3 says "It is NOT a ceiling in the sense of D1". It is: an upper clamp on a
    deposit, counted as destruction.
  - CLAUDE.md line 79 says "trace ADD edits through FieldEdit are additive with no
    ceiling (D1)". §12(b)'s amendment edits only the neighbouring clause ("No trace
    plane is clamped…"), so after P6 the row states both "no ceiling" and "bar the
    rail".
- **Is the rail justified?** Yes. The alternatives are a wrap, which turns a cell
  negative so the sweep reads it as clear and poison as nothing (an RL exploit), or
  a mid-tick raise. At 128 units it is 440× the real maximum. And it honours the half
  of D1 that mattered to Erik ("never cut the tile"): `min(s + add, MAX) ≥ s` for
  `s ≤ MAX`, so existing smoke is never cut.
- **Fix.**
  - In §5.3, replace "It is NOT a ceiling in the sense of D1" with: "It is D1's
    option (b) at the int32 representability limit, 128 units, 440× the real
    maximum. It never cuts smoke already on the tile, which was the substance of
    Erik's (a). It caps only an addition that cannot be represented, and books it."
  - P6 replaces the whole row sentence with: "No trace plane is clamped anywhere
    outside a reader, bar the counted int32 rail (`TRACE_CELL_MAX`, #88) at every
    site that adds to a cell; trace ADD edits through FieldEdit are additive up to
    that rail and never cut the tile (D1). Every loss is booked: vent in the trace
    slots of `boundary_flux_`, wipe, sink, decay, clip, and on the Python side
    `trace_pump_books` / `trace_deposit_books`. Other movers (vent plenum ledger, W3,
    `destroy_wall`/`seal_tiles`) conserve or book through those." Drop
    "already-booked" unless B2 is fixed.
  - The **Gas pump primitives** row gains: "trace is priced by air (#88 P0): the
    split runs over the bulk planes; each trace plane leaves at the cell's
    `S/N_bulk`."
  - The #88 closing comment says in one line that D1's cap now exists at 128 units,
    as information for Erik, not a question. His ruling is the one being bounded.

### B4. MAJOR: the re-baseline plan. Two golden events break the branch's own gate-h precedent, and P0 cannot lawfully move the golden at all

- **Evidence, P0.** The canonical scenario has no pump, vent or duct
  (`field_ab_harness.py:120-151`: smoke, fire, water, a wave, one marine, one breach).
  P0 changes only `extract_gas_n_vec` and its callers, so the golden cannot move. P0's
  row ("If the golden moves, re-baseline once with rationale") pre-authorizes an
  unexplained move.
- **Evidence, P3 + P5.**
  - The lineage in `tests/_xarch_perfield_digest.py:100-101, 111-112, 145` folds a
    branch's changes into "ONE re-baseline for the whole branch (the gate-h rule)",
    with every cause named in the one entry.
  - P3's value (k = 8 with ceil decay) is a state that never ships.
  - Reordering does not save a move: at k = 0, nearest and ceil already differ on the
    scenario's 0.6 seed (39 322 · 22 / 2^16 = 13.2, so ceil gives 14 and nearest 13).
- **Fix.**
  - P0's gate: **`GOLDEN_AGGREGATE` reproduced; a move is a bug.**
  - Fold P5 into P3: one flip patch, "`TRACE_FINE_BITS` 0 → 8 and decay ceil →
    nearest", with ONE golden event whose rationale carries the two-stage sole-cause
    proof:
    1. the k = 0 + ceil tree reproduces `f6ba810e…`;
    2. k = 8 + ceil gives an intermediate X, recorded in the entry and not committed
       as the golden;
    3. k = 8 + nearest gives the committed Y.
  - Each stage is reproduced twice. This keeps both sole-cause checks the doc wants,
    with one lineage event. If P3 and P5 must stay separate patches for the
    one-writer cadence, P3 may carry X on the branch, but the lineage entry is
    written once, at P5, naming both causes.

### B5. MINOR: list the gameplay changes with their numbers, not as "timing follows"

§6's "allowed to differ" table ends with "poison/teargas timing … follows from the rows
above". These are RL-visible, and each one should be a row with its size:
- **Poison's lethal window at the dose edge grows about 1.7×.**
  - Today, poison at 0.06 (3932 counts, `frac_q` 11) loses `ceil(0.66) = 1` count a
    tick. It reaches the 0.05 threshold (3277) in 655 ticks = 27 s.
  - At k = 8 it decays at the configured e-fold: `ln(1.2)/0.004` = 46 s.
  - This is intended (Erik's ruling), but it lengthens every thin poison cloud's
    damage time.
- **Teargas BLINDED near its threshold lasts about 20 % longer.** At 0.15 (9830
  counts, `frac_q` 27) today removes `ceil(4.05) = 5` counts a tick, against 4.05
  exact.
- **P0's pump and vent rates in smoke.**
  - Today a pump in smoke takes air at `want·N/(N+S)`; after P0 it takes the full
    `want`.
  - In the densest real smoke on ambient air (0.29 units over 1.0), that is +29 %
    air pumped and +29 % smoke removed.
  - Airlock `at_target` latches sooner in smoky rooms. Compressed tiles see less
    change.
  - This replaces D-A's "slightly".
- **D-B's diffusion change.** It is confined for poison and teargas: their
  densities are large in counts, so diffusion already acts above 240 old counts =
  0.0037 units per tile. The newly-live band sits below every dose threshold. Say so,
  so that a P3 reviewer does not look for a footprint change that should not exist.

### B6. MINOR: P5's floor of 1 makes the `frac_q == 0` skip load-bearing; put it in the kit

- Today `lost(v, 0) = (0 + 65535) >> 16 = 0` by itself.
- With `max(1, …)`, `lost(v, 0) = 1`, so a non-decaying trace gas would decay by one
  count a tick.
- Both callers skip it (`bulk_transport.cpp:665`, `cuda_bulk_transport.cu:951`), but
  the header only says "callers skip".
- **Fix:** `if (frac_q <= 0) return 0;` inside `trace_decay::lost`, plus a V7 case
  with decay = 0.
- **Also:** v2's V8 test (`test_smoke_transport_v2.py:401-424`) loops `range(v + 1)`
  ticks. At k = 8 that is 10^6 iterations as an upper bound, and the actual
  extinction of 0.06 poison takes about 38 000 ticks (1460 s exponential + 124 s
  linear). P5 rewrites the bound from the law (exponential to the linear edge, plus
  that many ticks), not from `v`.

### B7. MINOR: specify the clip books' window and sign

- `EOSSolver` books reset at `step` entry (`eos_solver.cpp:353-357`).
- Combustion runs after `run_substeps` (`physics_runner.py:939, 976`), so a
  `trace_clip_sum` written by the soot mint falls inside the same tick window. That
  holds only because of the order.
- `GameMap.trace_clip_books` is described as "Python ints", without saying whether
  it is per tick or cumulative.
- **Fix:** both books are per tick, reset at the same point, and signed ≥ 0. The doc
  names the window as the one `sim.step()` brackets, the window v2's identity tests
  already use. CUDA's slot is appended, never inserted (the P5c slot-12/13/14
  precedent).

### B8. MINOR: V5 should pin P0's books, not only its ratio

V5 checks the bulk equality and the ratio bound. Add:
- the pump's `gas_energy` book `pump_extract` equals `want·T_abs` exactly in smoke as
  in clear air;
- the vent plenum `trace_raw + sink` grows by exactly the trace removed;
- with B2: `trace_pump_books` equals the trace removed.

*Breaks if* the priced-by-air trace leaves the tile without landing in any book.

### B9. MINOR: §13 "Nothing here needs Erik" should carry B3's one line

The delegation covers the decision. The rail still bounds a ruling Erik made in his own
words, so the sentence should say: "the rail bounds D1 at 128 units; reported in the
#88 comment".

## 17. Critique — inventory & plan lens

Scope: an independent sweep of every reader and writer of the trace planes, read in the
code at `00e8019`. It covers `src/`, `cpp/src`, `renderer/`, `tools/`, `tests/`,
`main.py`, levels, the recorder, saves, the entity serializer and the integer reference.
It then checks the Systems section, the patch plan and the build path on the Home Desktop.

**Confirmed sound (no finding):**
- **C++ inventory is complete.** These are the only TUs that take the gas array:
  `bulk_transport`, `combustion`, `physics_engine`, `eos_solver`, `radiation_sweep`,
  `sky_exchange`, `bindings` and their CUDA twins (`cuda_bulk_transport`,
  `cuda_combustion`, `cuda_eos_step`, `cuda_eos_resident`, `cuda_kick_compression`,
  `cuda_radiation_sweep`).
  - Every density shift on a trace value goes through the FP_HD finishes
    `gas_extinction_finish` / `gas_extinction_exp_finish` (`radiation_sweep.h:248-277`).
    The CPU sweep, the CUDA pre-pass (`cuda_radiation_sweep.cu:262-318`, glow included)
    and the hover binding all call them, so changing those two bodies converts every
    sweep site on both backends at once.
  - `fire_simulation.cpp:98` and `cuda_fire.cu` take `smoke` and never touch it. The
    binding comment "emission scatter" at `bindings.cpp:1037` is stale but harmless.
  - The other `>> FP_SHIFT` sites in the trace TUs are rate shifts on a count, so they
    are scale-free: diffusion `bulk_transport.cpp:758`, `cuda_bulk_transport.cu:823`,
    decay `trace_decay.h:44`, and W3's `mul_q16` at `physics_engine.cpp:1120-1129`.
  - No hidden `65536` is applied to a trace value in C++.
- **Saves, levels, replay.** No save system exists: `SimState` holds references
  (`simulation.py:154-175`), and the only gas copy is the hot-reload round trip
  (`gamemap.py:1792-1796`). No `levels/*/level.toml` stores a trace amount. The vent
  entity serializes authoring fields plus the plenum ledger as `<q`
  (`entities/vents.py:72, 117-120`).
- **Old recordings.** The recorder records `smoke` only, dequantized to real units
  (`recorder.py:189-194`), and no caller passes a custom `fields` list that includes
  `gas`. Old `.npz` files stay valid, which agrees with §2.4.
- **Python gameplay readers** are the ones §2 names. All gas-amount writers
  (`payloads.emit_gas`, `combat.py:1357-1375` spray, `physics.py` smoke disc and clear)
  go through FieldEdit, and FieldEdit refuses a conservative slice
  (`field_edit.py:614-625`).
- **Systems reuse** is right in its main lines:
  - the FieldEdit door stays the one door;
  - the recorder keeps the real-unit float contract, so the DTYPE rule does not trigger;
  - the digest takes a version bump;
  - the gates run through the CUDA harness;
  - coefficients stay Q16, so the gases door's headroom argument is untouched.
  No parallel system is created.
- **Build path.** `cpp/build_cpu_home.bat` and `cpp/build_cuda.bat` both `cd /d
  "%~dp0\.."`, so they build into the worktree that holds the script. The suite and
  `cuda_harness` resolve `ROOT/cpp/build/Release` and `ROOT/cpp/build_cuda` relative to
  the test file. This worktree already holds both modules (built 2026-10-20 07:40).

### C1. MAJOR: the test fixtures have no inventory, and the silent ones bypass V1's lint

- **Evidence.**
  - P3's row says "about 50 files, §2.4's test list in the inventory". §2.4 has no test
    list.
  - 85 test files touch a trace plane, and 43 write one directly.
  - The dangerous writers are the ones V1's deleted-name lint cannot see: they get
    256 times thinner at k = 8 and keep passing.
- **The three silent classes**, each with one example:
  1. **Local literals.** `Q = 65536` in `test_smoke_transport_v2.py:48` and
     `cuda_trace_smoke_check.py:69` (the v2 transport's own gates), and
     `FP_ONE = 1 << 16` in `tools/fire_timing_harness.py:98`, whose smoke "mass" at
     `:504-506` keeps a `mass > 1.0` raw-count threshold.
  2. **The wrong boundary module.** `test_light_field.py:71` seeds `g.smoke` with
     `fire_fixed.quantize_scalar(0.4)`. `fire_fixed` keeps Q16, so nothing breaks
     loudly.
  3. **A second reference vocabulary.** `tests/_radiation_sweep_harness.py:114-132`
     `random_gas` draws its trace densities in Q16 (`Q(0.27)`, `ONE`). Its users are
     gate 0 (`test_radiation_sweep_reference.py`), `cuda_radiation_sweep_check.py`,
     `_cones_check`, `_light_check`, `test_radiation_sweep_light.py`,
     `test_radiation_sweep_cones_sky.py` and `test_sweep_fine_currency.py`. §7 and A8
     fix only `sweep_ref_q_gates._gas_planes`. At k = 8, `random_gas` makes gate 0 and
     the CUDA sweep checks compare two near-transparent media: equal, and vacuous.
- **Fix.**
  - Commit the fixture inventory as a list in §2: file, line, class, conversion.
  - Move every fixture to `quantize_trace` / `TRACE_ONE` in **P1, at k = 0**, where the
    move is an identity. Any test whose outcome changes at k = 0 has then been
    rewritten wrongly, which is a real oracle for the rewrite.
  - `random_gas` takes the A8 treatment: `trace_from_bulk(v) + r` on its trace planes,
    with the bulk pair untouched. Every gate that uses it prints its count of absorbing
    cells, as §7 demands of the reference's own gates.
  - Add a standing lint (V1): any assignment or `+=` into a trace slice
    (`gas[<trace id>]`, `.smoke`, `gas[:N_TRACE_GASES]`) whose right-hand side does
    not come from a `trace_*` name fails. The lint covers `tests/` and `tools/` as well
    as `src/` and `renderer/`.
- **Also a mixed-currency oracle in tests.** It belongs in §2.2's table.
  - `test_ps1_smoke_roundtrip.py:119-122` asserts that `Σ` over ALL planes never rises.
  - At k = 8 the soot mint takes `soot` bulk counts from N2 and adds `soot << 8` to
    smoke, so the total rises by `255·soot` and the test fails at P3. The tempting
    "fix" is to bend it.
  - The right property: `Σ bulk` never rises, and the trace planes close by their books
    (V6).

### C2. MAJOR: V3 tests reader functions, so a trace site renamed to a bulk name passes V1, V2 and V3

- **The mutation.** Deleting the ambiguous names forces a choice at every call site,
  but a mechanical rename can make the wrong one, e.g.
  `game_renderer.py:486 gas_fixed.dequantize_f32(gmap.gas[:N_TRACE_GASES])` →
  `dequantize_bulk_f32(...)`.
- **Why every gate misses it.**
  - V1 sees no deleted name.
  - V2 is an identity at k = 0.
  - V3 as written checks the conversions (`dequantize_trace_f32`, `trace_density_shr`,
    …) on hand-built planes, not the sites that should call them.
  - The same hole exists at `hover_readout.py:161-164` (the mixed dict),
    `game_renderer.py:479, 510, 578-579`, `sensor_accessor.py:130-132`,
    `exchange.py:625, 670`, `combat.py:1157` and `recorder.py:190-194`.
- **Fix.**
  - V3 drives the **public entry points** on a real GameMap:
    `pack_hover_readout`, the sensor accessor's `sample` and `area`, the teargas and
    poison rows of `exchange`, `fire_beam`, `PhysicsRecorder.record`, the engine's
    sweep (one tick, reading `a_eff` / light / glow through the binding), and the
    renderer's trace dequantize.
  - Fold the renderer's four trace-dequantize sites into one helper (C9), so V3 can
    call it without raylib.
  - V3's readers are a **registry table** in the test. A subset lint asserts that every
    `src/` / `renderer/` module that reads a trace slice (the C1 regex) appears in it.
    This is a classification check, not a count: a new reader fails until it is
    registered, and the registry may grow freely.
  - Add a one-line lint: no `bulk_*` / `quantize_bulk*` / `dequantize_bulk*` call takes
    a trace slice.

### C3. MAJOR: P1's oracle is blind by construction, and the plan pushes the misses into the flip patch

- **Evidence.**
  - P1's tier note admits that "the oracle cannot see a missed conversion at k = 0,
    hence V3 in P3".
  - So every missed conversion surfaces in P3 and is fixed there.
  - That breaks §6's "the flip is one constant", and with it the VERIFIED-SOLE-CAUSE
    argument the re-baseline rests on.
  - The golden scenario also never exercises poison, teargas, the laser, sensors, vents,
    pumps or W5 (`field_ab_harness.py:120-151`). Even a k ≠ 0 golden would leave those
    readers dark.
- **Fix: a "dry flip" in P1's gate.**
  1. After the k = 0 golden reproduces, the P1 agent builds the same tree at
     `TRACE_FINE_BITS = 8` locally, uncommitted, on CPU and CUDA.
  2. It runs V3, V4, the C1 lints and every reference / `random_gas` gate's
     absorbing-cell count.
  3. Any failure is a P1 bug, fixed in P1, and the dry flip is re-run.
- **Result.**
  - P3's diff becomes literally the four twins, spec v7, the golden entry, and P5's
    one line if B4's fold is taken.
  - The full suite at k = 8 stays P3's gate. Its failures are then behaviour changes
    (decay, diffusion), never conversion misses.
  - **P1's gate gets a real mutation-catcher:** a deleted kit call at any registered
    reader fails the dry-flip V3.

### C4. MAJOR: CUDA gates SKIP green when the worktree build is missing, and they pass on a stale build

- **Evidence.**
  - Every `test_cuda_*` wrapper is `pytest.mark.skipif(not cuda_available())`
    (`test_cuda_trace_smoke.py:34`).
  - `cuda_available()` only checks that `cpp/build_cuda/breach_physics*.pyd` exists
    (`cuda_harness.py:43-87`).
- **Two ways a gate goes vacuous.**
  - A failed or absent CUDA build makes every CUDA gate in §8 and §11 (P1 "every CUDA
    check", P2's rail test, P4) **skip**, and `pytest tests -q` reports green.
  - A C++ change rebuilt only on CPU leaves `build_cuda`'s module stale. That module
    carries its own copy of the CPU code too, so "GPU == CPU at tol 0" compares stale
    with stale and passes.
- **Ambiguity in the plan.** The "mode" column says "worktree subagent", but §11 says
  "every patch runs in this branch's worktree".
  - An `isolation: "worktree"` agent is cut off MAIN (the memory note
    `feedback_agent_worktree_base_branch`).
  - That worktree also has no `cpp/build` and no `cpp/build_cuda`, so it is the skip
    case above.
- **Fix: state in §11, for every C++-touching patch:**
  1. It runs in `C:\Users\steen\projects\breach\.claude\worktrees\88-smoke-unit-rescale`
     itself, with no `isolation`, one writer at a time.
  2. It runs `cpp\build_cpu_home.bat` (which is `--fresh`, so it is a full rebuild) and
     `cpp\build_cuda.bat` (Ninja, incremental; CUDA_PATH is v12.4 on this box), and
     records both `BUILD_EXIT=0` lines in the report.
  3. It checks that both `.pyd` files are newer than the newest `cpp/src` file.
  4. It runs the CUDA gates with `pytest tests -q -rs -k cuda` and reports **zero
     skips**. A skip is a gate failure.
- **And make V1 bite on the GPU module:** run the twin guard inside `run_cuda_script`
  against the CUDA module as well. A stale `build_cuda` then fails V1 at the flip.

### C5. MINOR: the "#78 pattern" is misquoted; the literal Python twin is right, so say why

- `optics_fixed.py:85-110` holds **no copy** of `E_FINE_BITS`: it reads `fine_bits` from
  the bound engine, and returns **0 when no engine is bound**.
- Copying that pattern for the trace currency would be a trap. GameMaps without an engine
  (level load, most unit tests, the tools) would silently read the trace planes at
  k = 0.
- **Fix.** §4 says "a literal twin, unlike #78's read-through, because `gas_fixed` is
  import-light and serves engine-less GameMaps". V1 holds the twin equal on both
  modules (C4).

### C6. MINOR: the digest spec's CHANGE PROCEDURE must name the currency

- `field_digest_spec.toml:6-9` triggers a version bump on "dtype/shape/membership/order"
  only.
- v7 adds `trace_fine_bits`, but the next rescale (k = 9, or a bulk change) would
  follow the written procedure and skip the bump.
- **Fix.** P3 amends that comment: "…or a field's integer CURRENCY (`trace_fine_bits`)".

### C7. MINOR: "rail" collides with the gas-energy rails

- CLAUDE.md's Gas energy recovery row and the resident code already use "rail" for the
  recovery's write-back (`e_rail_sum`, the "rail" book block at
  `cuda_eos_resident.cu:38, 933`).
- A second meaning in the neighbouring Trace transport row invites a counter in the
  wrong block.
- **Fix.** Use the name the books already carry: kit `trace_add_sat`, books
  `trace_clip_*`, and "the counted int32 saturation" in the rule. Or, at minimum,
  "trace rail" everywhere and never a bare "rail".

### C8. MINOR: inventory misses outside §2

All of these are bulk or scale-free. The `gas_fixed` users among them fail loudly at P1
under the deletion, but P1's scope should count them:
- **`vent_system.py`.**
  - `:322` the vent rate, a bulk count.
  - `:371` `t_min` / `t_max`, which are temperatures (A12's Q16 name).
- **`hover_readout.py`.**
  - `:176-179` `bulk_n` through `_GAS_FP_ONE_F`.
  - The `gas_energy` display divides by `_GAS_FP_ONE_F ** 2` (docstring `:45`). It
    must become `BULK_ONE ** 2`, never `TRACE_ONE`.
- **`gamemap.py:2644`**: a non-zero test, scale-free (add it to §2.5).
- **Tools.**
  - `tools/plot_sealedbox_70.py:40` and `tools/sky_exchange_bench.py:44`: bulk.
  - `tools/bench_two_room.py:96-97`: raw smoke bytes, scale-free.
  - `tools/fire_timing_harness.py:504-506`: C1 class 1.
- **Tests that mix currencies but stay exact** (no change, listed so that P3 does not
  "fix" them): `test_vent_mechanism.py:81-87` and `test_b4_pump.py:386-389`.

### C9. MINOR: one renderer seam for the trace dequantize

- `game_renderer.py:479, 486, 510, 578-579` each dequantize a trace slice inline.
- The Dequantize-convention rule wants that done once at `upload_state`.
- **Fix.** Add one `_trace_floats(gmap)` (the `(N_TRACE, h, w)` float32 block) that the
  four sites slice. It is the call site V3 drives (C2) and the single place a future
  resident gather swaps in.

### C10. MINOR: V6 must enumerate the rail sites, per backend

- §5.3 lists seven rail sites. V6's composite scenario hits only three: FieldEdit, 3b
  and the soot mint.
- The CUDA rail test covers 3b only.
- **Fix.** Parametrize over §5.3's rows on CPU: FieldEdit, 3b, mint, W5, the W3
  receiver, `seal_tiles`, `_inject_gas_n_core`. On CUDA: 3b chained, 3b resident, and
  the mint. Each case asserts plane ≤ `TRACE_CELL_MAX`, plane ≥ 0, and `clip` = the
  exact excess. *Breaks if* any single site wraps, raises or drops unbooked. The
  composite cannot say which.

### C11. MINOR: the patch order, tiers and splits

- **Merge P4 into P3.** After P3 re-baselines `GOLDEN_AGGREGATE` on CPU, every CUDA
  consumer listed in §8 is red until P4, so the branch carries a known-red state.
  The same writer already holds both builds. With B4's fold, that gives one flip patch:
  k 0→8, decay ceil→nearest, CPU and CUDA, one lineage entry.
- **P2 → Opus, not Sonnet.** Its oracles see only the planted scenarios (C10). The
  placement calls are judgement: once per cell on the net, not per face (A4); the book
  windows (B7); appended CUDA slots; the raise-free mutation pass in `seal_tiles`
  (`gamemap.py:2751-2755`); B1's mint and deposit books.
- **P3 tier.** With C1 and C3 taken, P3 is a mechanical flip under oracles, and Sonnet
  fits. Without them, keep Opus.
- **P0 → Opus and P1 → Opus** are right.

### C12. MINOR: each gate's mutation, caught and missed

| gate | catches | misses (and the fix) |
|---|---|---|
| P0 V5 | trace inside the bulk budget; trace priced against `N_total` instead of `N_bulk` (the bound names `N_bulk`) | the taken trace landing in no book (B8); the `N_bulk == 0` divide (A5) |
| P1 V2 (golden at k = 0) | any semantic change at k = 0: FieldEdit's float path, the door refusing a shipped row, the reference's arithmetic | every missed or doubled conversion; a trace site renamed to bulk (C2, C3) |
| P1 V1 | a twin moved alone; a deleted name used again | a wrong-currency choice (C2); a stale CUDA module (C4) |
| P2 V6 + golden | a wrap or raise at the planted sites; a rail binding below the limit (the golden would move) | the sites not planted (C10); a clip with no closed identity to stand on (B1) |
| P3 V3 / V4 | as specified: a mis-scaled conversion function | a call site that skips the function (C2); a fixture gone thin (C1) |
| P4 V9 | GPU ≠ CPU | a mis-scaling inside a shared FP_HD kit function, which is identical on both backends by construction: only V3 sees it. Say so in §8 |
| P5 V7 | a rounding regression back to the linear regime | nothing in its lens (B6 adds the `frac_q = 0` case) |

### C13. MINOR: verification wording against the test standard

- **V1's "the four §2.4 INLINE C++ sites call the kit"** pins a hand list. Assert the
  property instead: the two `radiation_sweep.h` finishes contain no `FP_SHIFT` and do
  call `trace_density_shr`. Every C++ trace reader reaches the shift through them (see
  "Confirmed sound").
- **V2 is a report, not a standing test.** That is correct, and it needs no "breaks if"
  line.
- **The remaining V-items** name a property and the change that breaks it. None of them
  pins a count or a set.
