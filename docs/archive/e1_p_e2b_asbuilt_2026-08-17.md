# P-E2b as-built — deposit dial + inventories (energy-books arc, 2026-08-17)

**Status: as-built record for P-E2b (design `energy_transport_design_2026-08-16.md`
v2.2, §2.2/§2.4/§2.5 + §6 row 5). Branch `storm-damping`, base `8435bb5` (the
P-E2a as-built docs commit). CPU **and** CUDA in the SAME commit. Code commit:
`d8af9d2`.**

---

## 0. Headlines

1. **`n_floor_heat` moved to a low value-hygiene dial: 0.05 → 0.01.** Rewritten
   at both config sites (`config.toml`, `temperature_solver.h`) to state the new
   role and retire the old 0.2-trial warning, which is moot in the downward
   direction.
2. **Building the required verification probe found a real overflow bug, and it
   is fixed at both deposit sites, CPU and CUDA.** The old two-step
   `mul_q16(deposit, recip_n)` → `recip_mul(·, recip_cv)` chain narrowed
   `deposit/N` to Q16.16 int32 (~32768 ceiling) *before* dividing by `c_v`. At
   `n_floor_heat = 0.01` a routine ~330 per-tick deposit divided by the floor
   alone is already 33,000 — past that ceiling. The narrow silently wraps
   (typically to a large negative number), and `heat_saturating_add`'s
   `delta <= 0` early-return then **drops the entire deposit — no clamp, no
   counter, nothing**. This is exactly the "temperature not backed by energy"
   failure class the arc exists to close, arising at the deposit SOURCE. Fixed
   with a new shared wide helper (`deposit_dT_wide_q16` / its CUDA twin) that
   chains all three factors through one 128-bit product and narrows once, to
   int64, with the caller clamping to `[0, INT32_MAX]` before the final Q16.16
   narrow. See §2 for the full account; this was **not** anticipated verbatim
   by the brief but is squarely inside its literal instruction ("make the
   reciprocal path int64 ... without precision collapse").
3. **`e_deposit_drop_sum`: both deposit sites now carry an energy-sum counter**,
   CPU + CUDA, bit-identical cross-backend, non-vacuous in the CUDA lockstep
   sweep.
4. **`n_work_ref` is plumbed and proven inert**: bench digests byte-identical
   with the dial overridden to 999.0 vs the 0.25 default.
5. **The deposit-dial change (0.05 → 0.01) measures as behaviorally INERT on
   every live scenario in this repo** — ledger, `fire_tune_loop` scorecard, the
   flame-cell histogram, and the one live ignition-timing test all report
   *zero* difference between the two floor values, because flame-zone gas
   density never drops anywhere near either floor in these scenarios (measured
   mean n_bulk ≈ 0.43, p0 ≈ 0.39). This is a reported measurement, not a
   defect — see §5.
6. **Suite: 48 failed / 2173 passed / 5 skipped / 1 xfailed, set-diff EMPTY**
   against the identical baseline. No test moved.
7. **§5 T-threshold consumer inventory executed** (§4 below). Two live,
   currently-unguarded gas-temperature consumers found and **reported, not
   fixed**: the EOS CFL sound-speed max-reduction, and the level-designer
   `temperature` area-mean sensor. A third, cosmetic class (render fire-light
   selection) is also reported.

---

## 1. `n_floor_heat` — the low dial (design §2.2)

### 1.1 What changed

| site | before | after |
|---|---|---|
| `config.toml` | `n_floor_heat = 0.05` | `n_floor_heat = 0.01` |
| `cpp/src/temperature_solver.h` | `float n_floor_heat = 0.05f;` | `float n_floor_heat = 0.01f;` |
| `src/simulation/physics_runner.py` | fallback default `0.05` | fallback default `0.01` |

Both `[physics.thermal]`'s rationale block and the `[physics.combustion]`
shared-dial note (`config.toml`) are rewritten to state: the floor's stability
job is gone (P-E1 closed the EOS transport books; T_MAX_PHYS is the value
backstop now); the old 0.2-trial warning ("measurably perturbed marginal
ignition timings") is **moot in this direction** — a *lower* floor deposits
*more* into thin cells, so marginal ignition trends faster, not slower; the v1
"0.25 shared with the trust gate" ruling is retired (n_work_ref is unrelated
and separately plumbed, §3).

### 1.2 The reciprocal-path verification, and what it found

The design asked for the reciprocal path to "get int64 intermediates so even
0.001 is reachable without precision collapse," to be verified by a probe. I
built `tools/e2b_floor_reciprocal_probe.py`, calling the **actual C++
primitives** both deposit sites use (`fp_reciprocal_q16`, `fp_recip_mul`, new
debug bindings added this patch) rather than a re-derived Python
approximation, and swept `n_floor_heat` down to 0.001 against both a "typical"
(330, the P-E1 as-built's own measured per-tick deposit reference) and a
"stacked firestorm" (2600, the config's own historical worst-case reference)
deposit magnitude.

**First run crashed.** `bp.fp_recip_mul` raised
`TypeError: ... invoked with 2163882600, 4294967296` the moment the sweep
reached `floor=0.01` — the Python mirror, replicating the *existing shipped*
C++ arithmetic exactly, produced an intermediate (`deposit/floor`, in Q16.16)
that had already overflowed int32. This is not a probe bug: `reciprocal_q16`
itself (already int64-internal, self-guards to a floor of 3 raw counts) was
never the problem — the problem is the SUBSEQUENT `mul_q16` narrow, which
forces the intermediate `deposit/N` value into Q16.16's representable range
(**magnitude ≤ ~32768**) before the `/c_v` step ever runs. At `floor = 0.01`,
`330/0.01 = 33,000` — already past that ceiling. `heat_saturating_add`'s
`delta <= 0` early-return (raycaster.h:69-77) then silently drops the deposit.

**Was this already possible at the shipped 0.05 default?** Yes, for the
"stacked firestorm" magnitude: `2,600/0.05 = 52,000` already overflows. The
config's own historical comment believed T_MAX_PHYS would bound this case —
it can't, because the overflow happens in an intermediate *before* T_MAX_PHYS
ever sees the value. The 0.01 ruling makes the SAME bug reachable by a
*routine* single-fire deposit instead of only an extreme stacked case — which
is exactly why the design anticipated this and named it explicitly.

**The fix** (`cpp/src/fixed_point.h`'s `deposit_dT_wide_q16`, CUDA twin
`deposit_dT_wide_q16_dev` in `cuda_fixedpoint_device.cuh`): chain
`deposit * recip_n * recip_cv` as one 128-bit product (mirroring `recip_mul`'s
existing `__SIZEOF_INT128__` / MSVC `_mul128` / portable dual-path structure),
narrow **exactly once**, to `int64_t` — never to `q16`. Both call sites
(`combustion.cpp`'s gas branch, `temperature_solver.cpp` Pass 1's gas branch,
and their CUDA twins) now clamp this wide result to `[0, INT32_MAX]` *before*
the final narrow for `heat_saturating_add`. An honestly-huge deposit still
hits the counted T_MAX_PHYS rail exactly as before — it just arrives through a
value that was never corrupted on the way there.

The `e_deposit_drop_sum` energy-counter calculation (below) had the identical
latent bug (it also computed `deposit/floor` via the narrowing chain) and is
fixed the same way, using plain-int64 staged arithmetic (`mul_wide(...) >>
FP_SHIFT` then one more `>> FP_SHIFT`) rather than the 128-bit helper, since
its own factors stay within safe int64 bounds without it.

**Probe result, post-fix (`tools/e2b_floor_reciprocal_probe.py`):**

```
PROBE RESULT: PASS -- the WIDE reciprocal path (reciprocal_q16 ->
deposit_dT_wide_q16, int64/128-bit throughout, no premature q16 narrow)
stays arithmetically sane down to floor=0.001 for deposits far beyond the
stacked-firestorm reference; lowering the floor deposits MORE into thin
cells at every step of the sweep (the design's predicted inversion,
confirmed at the primitive level).
```

Representative sweep values (combustion site, deposit = 330 — the "typical"
reference — at N = 0.0005, below every floor in the sweep):

| floor | dT (raw) | dT (game-deg) |
|---:|---:|---:|
| 0.05 | 432,510,870 | 6,599.6 |
| 0.01 | **2,147,483,647 (clamped at INT32_MAX)** | 32,768.0 |
| 0.001 | **2,147,483,647 (clamped at INT32_MAX)** | 32,768.0 |

Even the "typical" single-fire deposit, once the floor drops to the shipped
0.01, is already large enough at this N to hit the wide-arithmetic ceiling —
the exact case the OLD chain silently wrapped to garbage on. Post-fix, it is
correctly recognized as enormous and clamped at `INT32_MAX` (raw), from which
`heat_saturating_add` + the explicit `T_MAX_PHYS` compare take over as
designed — not silently zeroed. At the OLD 0.05 floor the same deposit stays
comfortably unclamped (6,599.6 game-deg, an honest large-but-representable
value); the "stacked firestorm" (~2600) reference deposit clamps at
`INT32_MAX` at every floor in the sweep, including 0.05 — confirming the
overflow was already latent, just harder to trigger, at the pre-arc default.

**Deviation from the brief, flagged prominently per the workflow's "report,
don't hide" norm:** the brief scoped this patch as a "counters+parity oracle"
patch with the inventory being the only judgment-heavy piece. This overflow
fix is a genuine arithmetic correctness change at both deposit sites (CPU and
CUDA), reachable in principle at the *pre-arc* 0.05 default too for extreme
deposits. It is squarely inside item A's literal instruction ("the reciprocal
path gets int64 intermediates ... verify by sweeping the dial ... showing the
deposit arithmetic stays sane") — I did not go looking for a bug outside that
scope, and did not retune anything. But it changes what happens to an
extreme/starved-cell deposit (previously: silently dropped; now: correctly
clamped and counted), so it is reported here explicitly rather than folded
silently into "the dial changed." Measured to be invisible on every bench/
scorecard scenario in this repo (§5) — the fix only bites in a regime none of
the current benches reach.

---

## 2. `n_work_ref` — trust-gate dial, PLUMBING ONLY (design §2.4)

Added, per the `T_MIN` / `_ep("T_MIN", ...)` idiom:

- `cpp/src/eos_solver.h`: `float n_work_ref = 0.25f;` member, next to
  `T_WORK_CLAMP`.
- `cpp/src/bindings.cpp`: `.def_readwrite("n_work_ref", &EOSSolver::n_work_ref)`.
- `src/simulation/physics_runner.py`: `self.eos.n_work_ref = _ep("n_work_ref",
  self.eos.n_work_ref)`.
- `config.toml` `[physics.eos]`: `n_work_ref = 0.25`.

**Nothing reads this member anywhere** — not in `eos_solver.cpp`, not in any
CUDA kernel. The fade mechanism (fade factor = 0 below `n_work_ref/2`, linear
to 1 at `n_work_ref`) is explicitly P-E4's territory; this patch's only job is
config key → C++ member → bindings → Python bind.

**Inertness, measured** (`tools/bench_two_room.py --ticks 240 --pf1b`):

| run | all 7 field digests (traj + final) |
|---|---|
| default (`n_work_ref = 0.25`) | byte-identical |
| `--set physics.eos.n_work_ref=999.0` | **byte-identical to the default run** |

Overriding the dial to a wildly different value changes nothing — the plumbing
is provably inert.

---

## 3. `e_deposit_drop_sum` — the energy-sum counter at both deposit sites (design §2.2/§2.5)

Per the design ("`heat_floor_hits` exists only at the combustion site today;
both sites get energy-sum twins") and this patch's explicit item C, both
deposit sites now carry an `e_deposit_drop_sum` counter, in the SAME currency
as `deposit`/`heat` (Q16.16, single power — not the transport pass's
Q16.16² energy-books currency):

| site | what it counts | mechanism |
|---|---|---|
| `combustion.cpp` (`CombustionSolver::e_deposit_drop_sum`) | the floor-engagement drop: `deposit*(1 - n_real/floor)` when `N_total < floor` | new counter (L1-8's "starved-fire destruction" concern, DISSOLVED at 0.01 per the design ruling — measured drop is 0 on every bench run, §5) |
| `temperature_solver.cpp` Pass 1 (`TemperatureSolver::e_deposit_drop_sum`) | L3-7's attenuation drop: `deposit*(1 - min(N,1))` below ambient density — PHYSICAL, stays, never had a counter before | new counter |

Both accumulate across `step()` calls (never reset) — the `heat_floor_hits`/
`t_max_phys_hits` idiom this class already uses, matching P-E2a's own choice
for its six conduction counters (the site determines the idiom, per the
brief's instruction; both deposit sites already used the accumulate class).

**CUDA twins.** `combustion_pass_c` gains a dedicated `unsigned long long*
d_dep_drop` int64 atomic slot (separate from the existing 32-bit
`d_heat_floor_hits`/`d_t_max_phys_hits` pair, which are safe as int32 hit
*counts* but this is a value *sum*). `temp_convert_unified` grows a new slot
`C_DEP_DROP = 6` in the existing 6-slot `cnt` block (now 7 slots;
`TEMPERATURE_ENERGY_SLOTS` 6→7 in `cuda_temperature.h`), folded by
`physics_engine.cpp` into `this->temperature.e_deposit_drop_sum`.

**Python/binding surface grown, all call sites updated:**

- `bp.cuda_combustion_step` returns a 3-tuple now (`heat_floor_hits,
  t_max_phys_hits, e_deposit_drop_sum`) — `tests/cuda_combustion_check.py`
  (2 call sites) and `tests/cuda_po2b_check.py` (1 call site) updated to
  unpack and compare the third element.
- `bp.cuda_temperature_step`'s isolated GPU entry returns an 8-tuple now
  (`t_max_phys_hits` + 7 energy counters) — `tests/cuda_conduction_check.py`,
  `tests/cuda_thermal_mass_check.py`, `tests/cuda_cool_shift_check.py` all
  extend their `E_COUNTERS` tuple to include `"e_deposit_drop_sum"` (the
  three lockstep gates that call this entry, per P-E2a's own §8.1 finding
  that it has three callers, not one).
- `tools/storm_ledger.py`'s `counters()` gains `comb.e_deposit_drop_sum` and
  `temp.e_deposit_drop_sum`.

**CPU↔CUDA parity, measured** (`tests/test_cuda_p69_combustion.py`,
`tests/test_cuda_p66_conduction.py`, both direct-run via pytest):

- Combustion: "all edge configs + 15 fuzz cases (pinned span) + 15 fuzz cases
  (shipped span) bit-identical on gas/temperature/wall_hp + rail counters" —
  the 3-tuple including `e_deposit_drop_sum` compared exactly.
- Conduction/temperature: **"all 121 configs bit-identical on `temperature` +
  rail hits + the seven P-E2a/P-E2b energy counters"**, with explicit
  non-vacuousness: `e_deposit_drop_sum` Σ|per-config| = **262,768,893,329**
  (the `thin=True` synthetic sweep genuinely exercises N below the floor).
- Both files' ONLY failing leg is the pre-existing shared canonical golden
  comparison (`8203584350ae69a5...` — byte-identical to what P-T0 left it at;
  **not moved by this patch**, same reason P-E1/P-E2a found: the canonical A/B
  scenario carries `temperature` identically 0 on every cell/tick, an exact
  no-op for a law that only changes what happens to a nonzero deposit).

---

## 4. The §5 T-threshold consumer inventory (design §5, task item D)

Walked every read-side consumer of a temperature threshold/comparison across
`cpp/src/*.cpp|*.cu|*.h`, `src/simulation/*.py`, and `renderer/*.py`, using the
committed L3 writer table (`energy_transport_critiques_round1_2026-08-16.md`
addendum) as the completeness oracle for the WRITE side and a broad grep sweep
for the READ side. Ruled per class:

- **SAFE — solid/object T** (thermal-mass-backed; never exposed to the
  gas-N-collapse problem).
- **SAFE — different field entirely** (not `temperature[]`; already outside
  the energy ledger by design).
- **THIRD CLASS — raw gas T, no N-guard** (reported, NOT fixed — a threshold
  change is feel-adjacent per CLAUDE.md).

### 4.1 Solid ignition checks — ALL SAFE

| # | site | what it does |
|---|---|---|
| 1 | `cpp/src/combustion.cpp:473` | per-claimant "not yet burning, `Tsnap[i] < ignition_temp`" gate — `i` is always a flammable (solid) tile |
| 2 | `cpp/src/cuda_combustion.cu:269` | CUDA twin of 1 |
| 3 | `src/simulation/combat.py:573` (`apply_temperature_ignition`) | the primary, level-wide ignition seed: `hot = flammable & has_fuel & (T >= ignition_temp)` — masked to `flammable` (solid) tiles by construction |
| 4 | `src/simulation/combat.py:566` | the re-arm/hysteresis half of the same gate |
| 5 | `cpp/src/fire_simulation.cpp:230` | the soft sustain-side ramp feeding the growth/death logistic — `T = temperature[i]` restricted to `flammable[i]` |
| 6 | `cpp/src/raycaster.cpp:900,:920` / `cpp/src/cuda_raycaster.cu:249-252` | the "warm emitter" gate: `thermal_solid[i] && temperature[i] >= t_emit_q` — explicitly AND'd with `thermal_solid`; the receiver-T read is reachable only inside the Kirchhoff `a_r>0` branch (air's `heat_atten==0`), so it structurally cannot read gas T |

**Verdict:** every ignition/emission decision in the engine is masked to
`thermal_solid`/`flammable` (object T, thermal-mass-backed) or gated by
Kirchhoff absorptivity (air is radiation-inert). None of these are exposed to
the gas-N-collapse problem. Design §5's "ignition stays temperature-based"
holds, and it is honest — solid T is always energy-backed via thermal mass.

### 4.2 Gas ignition / `fuel_gas` flashing — NO CONSUMER EXISTS

Grepped `fuel_gas`, `flash`, `flashover`, `flash_point`, `vapor`,
`explosive_gas` across the engine and simulation layer. `fuel_gas`
(`src/simulation/gases.py`) is a defined gas plane (advected, diffused,
vented) with **no combustion-trigger code anywhere** — no site compares
`fuel_gas` density or a gas cell's `temperature[]` against a flash-point
constant. `combustion.h:306` names "shoebox flashover" as an explicitly
deferred feel question; design §2.6 queues the "explosive redesign" (real
bulk-N₂ + ΔE injection) as future work. **This category is empty in the
current codebase** — the brief's "flashes the `fuel_gas` plane" scenario does
not exist yet to be either safe or unsafe.

### 4.3 Unit heat damage — SAFE (different field, not `temperature[]`)

`apply_environmental_damage()` (`src/simulation/exchange.py:237-360`, the
implementation `test_unit_heat_damage.py` targets) reads `gmap.heat`
(per-tick radiant-flux accumulator) and `gmap.rad_flux` (a positive-only
sensor plane explicitly documented as "NOT PART OF THE ENERGY LEDGER — no
solver reads it," `raycaster.h:397-412`), never `gmap.temperature[]`. It
derives `T_felt` from flux directly. **Not exposed to the gas-T problem** —
it is backed by a different, deliberately out-of-ledger quantity.

### 4.4 Rails/telemetry — mostly write-sites (excluded); two live THIRD-CLASS finds

Every `if (temperature[i] > t_max_phys_q) { ...; ++hits; }` site (`combustion.cpp:815`,
`temperature_solver.cpp:222/265/313`, `eos_solver.cpp:796/1529`) is a **write**
(a clamp + counter), not a downstream consumer, and is excluded per the task's
framing.

**THIRD-CLASS FINDING 1 — CFL/substep-count sound-speed reduction,
UNGUARDED.** `cpp/src/eos_solver.cpp:347-351` (+ CUDA reference twin
`cuda_eos_step.cu:170-174`):

```cpp
int64_t t_max_abs_raw = (int64_t)t_amb_q;
for (int i = 0; i < n; ++i) {
    if (solid[i] || is_vacuum[i]) continue;
    const int64_t t_abs = (((int64_t)s_eos_q * (int64_t)temperature[i]) >> 16)
                         + (int64_t)t_amb_q;
    if (t_abs > t_max_abs_raw) t_max_abs_raw = t_abs;
}
```

This feeds `c_local_q` (local sound speed), which caps the velocity estimate
and directly determines `n_sub` (the substep count) for the whole tick's
advection. **The guard is `!solid && !is_vacuum` only — no `n_bulk`/N-floor
check.** Post energy-books (`T = e/n_bulk`), a single thin-N cell with a
rounding-dominated T can dominate this MAX over the entire open-air field and
change the substep count for the whole tick. Contrast with `p*` in the same
function (`:598-601`, `p* = c·N·T_abs`), which IS N-weighted — the N in the
product cancels the thin-N reciprocal back down, benign by construction. This
reduction has no such weighting. **Reported, not fixed** — a substep-count
change is a determinism/performance-adjacent decision, not a bare threshold,
but it is squarely the "acts on a temperature not backed by energy" pattern
§5 names, and any fix belongs to whoever owns the CFL/substep design, not this
counters-and-inventory patch.

**THIRD-CLASS FINDING 2 — the `temperature` sensor's area-mean, UNGUARDED and
UNTESTED.** `src/simulation/sensor_accessor.py:154-175`
(`EntityFieldAccessor.area()`), reachable from any level's `temperature`
sensor with `area_m > 0` via `src/simulation/sensor_system.py:86-92`, wired to
a `Decider` logic node (`src/simulation/logic_nodes.py`). The single-tile
sensor path (`SAMPLE_FAMILY = SAMPLE_BODY`) deliberately avoids gas T — its own
docstring says why: *"a faced-air sample would read a plume-advected gas
value."* But `area()` is channel-agnostic: for `temperature` with `area_m > 0`
it computes an **unweighted arithmetic mean of raw `gmap.temperature[]` over
the surrounding open-air (gas) tiles**, contradicting the single-tile design's
own stated rationale, with zero N-weighting or N-floor guard. **Not currently
exercised by any test** (`tests/test_b3_sensors.py` only covers the
single-tile path for this channel) — the gap is unexamined in practice but the
code path is live and wire-able by a level author today. **Reported, not
fixed** — this is gameplay logic-node wiring, well outside a physics-counter
patch's authority, and a threshold change here is exactly the class of
decision CLAUDE.md reserves for a feel-adjacent HUMAN-TEST gate.

### 4.5 Render reads — cosmetic THIRD-CLASS finds, lower severity

| site | what it does | guard? |
|---|---|---|
| `renderer/fire_lights.py:74-75` | `candidate = temp_field >= threshold` — selects fire-light-source candidate tiles from the FULL, unmasked `sim.gmap.temperature` array (`main.py:537`), then brightest-K sorts them | none |
| `renderer/blackbody.py:197-210` | `_intensity_curve`: a soft/clamped glow ramp (implicit floor at `kelvin_glow_min`) feeding both the `HeatFieldOverlay` (T-key) and fire lights, over the same unmasked full field | none |
| `renderer/hover_readout.py:80` | cursor readout text (informational only, no threshold/decision) | N/A |

A noisy thin-N gas spike could, in principle, be preferentially selected as a
bright fire-light source ahead of genuinely energetic tiles. **Reported, not
fixed** — purely cosmetic (no gameplay consequence), and any visual threshold
change is explicitly feel-adjacent (CLAUDE.md).

### 4.6 Summary

| class | count | disposition |
|---|---|---|
| Solid/object T (thermal-mass-backed) | 6 sites | SAFE, no action |
| Different field (heat/rad_flux, not temperature[]) | 1 mechanism | SAFE, no action |
| Gas ignition / fuel_gas flash | 0 sites | not applicable — no such code exists |
| **Raw gas T, unguarded (THIRD CLASS)** | **3 findings** | **reported here; NOT fixed** — CFL max-reduction (`eos_solver.cpp:347-351`), `temperature` sensor area-mean (`sensor_accessor.py:154-175`), render fire-light selection (`fire_lights.py`/`blackbody.py`, cosmetic) |

No silent fixes were made to any of these. Per the task's explicit
instruction, a threshold change is feel-adjacent and is Erik's call.

---

## 5. Margin re-measure (design §2's boundary instrument, task item E)

`tools/e1_flame_nbulk_histogram.py`, same fixture/dials/seed as P-E1's own
measurement, re-run at the new shipped floor (0.01):

| statistic | P-E1 (floor 0.05) | **P-E2b (floor 0.01)** |
|---|---:|---:|
| mean n_bulk | 0.4346 | **0.4344** |
| p0 | 0.3954 | **0.3938** |
| p1 | 0.3992 | 0.3989 |
| p50 | 0.4447 | 0.4311 |
| p100 | 1.0000 | 1.0000 |

| bin | P-E1 share | P-E2b share |
|---|---:|---:|
| [0, 0.01) | 0.0000 | **0.0000** |
| [0.01, 0.05) | 0.0000 | **0.0000** |
| [0.05, 0.1) | 0.0000 | 0.0000 |
| [0.25, 0.5) | 0.9721 | 0.9717 |
| [0.5, 0.75) | 0.0267 | 0.0271 |

**Reading:** essentially unchanged. Flame-zone density sits at mean ≈ 0.43,
p0 ≈ 0.39-0.40 — comfortably above BOTH dials (0.05 and 0.01). Nothing
migrated toward either floor; the distribution's tiny shift (< 0.1% in the
mean) is run-to-run texture, not a floor effect. This directly explains §6's
zero-delta measurements below: the floor simply never engages at this
scenario's operating point, at either value.

---

## 6. Deposit-dial behavioral measurement, 0.05 → 0.01 (task GATE 3)

**Method:** `config.toml`'s `n_floor_heat` was temporarily set to 0.05,
`fire_tune_loop.py` and `storm_ledger.py` run (PRE), then restored to the
shipped 0.01 and re-run (POST) — a clean, isolated one-dial delta. Nothing
else was retuned.

### 6.1 `tools/fire_tune_loop.py` scorecard

| metric | PRE (0.05) | POST (0.01) | delta |
|---|---:|---:|---:|
| peak I | 0.724 | 0.724 | 0 |
| peak time | 2.31 min | 2.31 min | 0 |
| fire death | nan | nan | — |
| flame plateau T (game) | 388 | 388 | 0 |
| far-field T rise (game) | 0.0 | 0.0 | 0 |
| room N_total min | 1.000 | 1.000 | 0 |
| far-field X min | 0.2061 | 0.2061 | 0 |
| local X min | 0.1914 | 0.1914 | 0 |
| wall_hp at end | 12.23 | 12.23 | 0 |

**Zero measured delta on every metric.** No verdict changed.

### 6.2 `tools/storm_ledger.py` (pump-off command, 4800 ticks, P-F1b)

| pass | quantity | PRE (0.05) | POST (0.01) |
|---|---|---:|---:|
| eos | `eth_gas` | 291.9 | 291.9 |
| combustion | `eth_gas` | −403.1 | −403.1 |
| tail | `eth_gas` | 71.86 | 71.86 |
| `comb.heat_floor_hits` | | 0 | 0 |
| `comb.e_deposit_drop_sum` | | 0 | 0 |
| `temp.e_deposit_drop_sum` | | 0 | 0 |

**Zero measured delta; the floor never engages at either value on this
bench.** This is the L1-8 "starved-fire destruction" concern's DISSOLUTION,
measured directly rather than argued: the drop counter is 0, not merely small.

### 6.3 Ignition-chain timing

The brief cites a "P-R4 record: ~5.0 s touching / 11.2 s across a 1-tile gap."
Searched the tree for a live, reproducible source of that pair
(`docs/fire_recalibration_2026-08-02.md`, `test_fire_heat_source.py`) and
found: **only the 1-tile-gap ("air-separated") scenario is a live automated
measurement** — `tests/test_fire_heat_source.py::_chain_sim()` /
`test_full_chain_heat_ignites_air_separated_wood` (currently a strict-xfail
XPASS, i.e. a carried red already in the 48-red baseline, unrelated to this
patch — Appendix A territory, untouched). No "touching" ignition-timing test
exists in the current suite (a face-adjacent target ignites via conduction,
P-E2a's territory, not radiation). The cited "5.0s/11.2s" pair is a historical
P-R4-era record that predates several later recalibration arcs (cool_shift
axis, thermal-mass axis, P-F1a/b) and is not reproducible from the current
tree as a same-conditions pair.

**Measured instead** (ad hoc probe reusing `_chain_sim()`/`_hold_burner()`
directly, bypassing the xfail marker), at floor 0.05 vs 0.01:

| floor | ignited tick | ignited time | T at ignition |
|---:|---:|---:|---:|
| 0.05 | 83 | 3.458 s | 302.3 game |
| 0.01 | 83 | 3.458 s | 302.3 game |

**Zero delta, and this is expected, not a miss:** the target tile in this
scenario is `MAT_WOOD` — a `thermal_solid`. Its heat conversion uses the
OBJECT branch (`deposit >> heat_inv_shift`), which the design explicitly
leaves untouched ("Object branch ... untouched — always honest") and which
`n_floor_heat` never touches (there is no gas divisor to floor there). This
scenario is therefore structurally floor-immune, and the design's predicted
ignition-timing effect (§2.2's "marginal ignition gets slightly faster") can
only show up in a scenario whose ignition path runs through actual thin GAS
temperature — none of which exists as a live automated timing test in this
tree. The effect is real and confirmed at the arithmetic-primitive level
(§1.2's probe), but is invisible on every scenario-level measurement available
in this repo.

### 6.4 Reading

The design's own text predicted floor engagement would be "~zero at 0.01" —
confirmed, measured, at zero across the ledger, the scorecard, the histogram,
and the one live ignition-timing test. The dial change is a real, deliberate,
documented ruling; its behavioral footprint on the shipped benches is
measured to be exactly zero. The overflow fix (§1.2) is the part of this
patch that DOES change behavior — but only in the extreme/starved regime none
of these benches reach, which is precisely why it went unnoticed until the
probe exercised it directly.

---

## 7. Gates — full accounting

1. **Baseline suite** (captured first): **48 failed / 2173 passed / 5 skipped
   / 1 xfailed** — matches the brief's expectation exactly.
   `test_air_boundary::test_ambient_gate3_udamp_band_absorbs_reflection`
   present as the carried red (P-E1 as-built §8), left untouched.
2. **Builds:** `cpp\build_cpu_data.bat` and `cpp\build_cuda_lenovo.bat` both
   `BUILD_EXIT=0`. `n_work_ref` inertness: bench digests byte-identical with
   the dial default vs overridden to 999.0 (§2).
3. **Deposit-dial behavioral report:** §6 (ledger + scorecard + ignition
   timing, all zero-delta, explained). Nothing else was retuned.
4. **CPU↔CUDA tol 0 across the lockstep set.** Both deposit sites' CUDA
   twins verified directly: `cuda_combustion_check` (3-tuple rail compare,
   `e_deposit_drop_sum` included) and `cuda_conduction_check` PART 1 (121
   configs, seven energy counters bit-identical, `e_deposit_drop_sum`
   Σ|per-config| = 262,768,893,329 — genuinely exercised). Both files' only
   failing leg is the pre-existing shared canonical golden, unmoved (byte-
   identical to what P-T0 left it at — confirmed, not assumed).
5. **Suite set-diff vs baseline: EMPTY**, both directions —
   ```
   diff baseline_failed_sorted.txt post_failed_sorted.txt
   (no output — identical)
   ```
   48 failed / 2173 passed / 5 skipped / 1 xfailed, name-for-name identical.
   All prior determinism pins stay green (implied by the empty set-diff —
   none of `test_no_transport_mint`, `test_transport_delta_is_one_way_negative`,
   the four P-E0 scenario-determinism tests, moved). `test_no_rail_hits`
   stays xfail (owner P-E4, unchanged). The T-threshold consumer inventory
   (§4) is committed in this doc.

---

## 8. Deviations from the brief, summarized

1. **A real overflow bug was found and fixed at both deposit sites (CPU +
   CUDA), beyond what a pure "counters + parity oracle" patch would normally
   touch.** Squarely inside item A's literal instruction; reported prominently
   in §1.2 rather than folded silently into "the dial changed." Verified
   invisible on every scenario this repo currently benches (§5, §6) — the fix
   only bites in the extreme/starved regime the synthetic probe and the CUDA
   `thin=True` fixtures reach, not the shipped game scenarios.
2. **The brief's "P-R4 record: ~5.0s touching / 11.2s gap" ignition-timing
   pair could not be reproduced as a same-conditions measurement** — only the
   1-tile-gap scenario exists as a live test; the "touching" case is a
   conduction (not radiation) ignition path with no dedicated timing test.
   Measured what is measurable instead (§6.3) and explained why it reads
   zero-delta (the scenario's target ignites via the object branch, which
   `n_floor_heat` never touches).
3. **Combustion's `e_deposit_drop_sum` (like the pre-existing
   `heat_floor_hits`) is not folded back into `self.combustion`'s Python-side
   counters when the live game loop dispatches combustion to the GPU**
   (`physics_runner.py::_run_combustion`'s CUDA branch discards
   `cuda_combustion_step`'s return tuple entirely). This is a **pre-existing
   gap**, not something introduced or fixed by this patch — `heat_floor_hits`
   already had the identical limitation, and `e_deposit_drop_sum` inherits it
   by construction (same call site). Noted here because a future GPU-backend
   ledger reader should know this counter reads 0 on that path today.
4. **The T-threshold consumer inventory (§4) found three THIRD-CLASS
   consumers** (a CFL max-reduction, a level-designer sensor, and cosmetic
   render fire-light selection) that read raw, N-unguarded gas temperature.
   **None were fixed** — per the task's explicit instruction, a threshold
   change is feel-adjacent and is reported for Erik's decision, not
   implemented here.

---

## 9. Files touched

**Code (CPU):**
- `cpp/src/fixed_point.h` — the shared `deposit_dT_wide_q16` wide-arithmetic
  helper (the overflow fix, §1.2).
- `cpp/src/combustion.h` / `.cpp` — `e_deposit_drop_sum` member + the
  floor-drop calculation, rewritten onto the wide chain.
- `cpp/src/temperature_solver.h` / `.cpp` — `e_deposit_drop_sum` member + the
  Pass-1 attenuation-drop calculation; the `n_floor_heat` default + rationale
  rewrite.
- `cpp/src/eos_solver.h` — `n_work_ref` member (plumbing only).
- `cpp/src/bindings.cpp` — `fp_reciprocal_q16` / `fp_recip_mul` /
  `fp_deposit_dT_wide_q16` debug bindings; `e_deposit_drop_sum` readonly
  bindings (both classes); `n_work_ref` readwrite binding; the
  `cuda_combustion_step` / `cuda_temperature_step` isolated-entry tuple
  growth.
- `cpp/src/physics_engine.cpp` — the 7th temperature energy-counter slot
  fold.

**Code (CUDA):**
- `cpp/src/cuda_fixedpoint_device.cuh` — `deposit_dT_wide_q16_dev`.
- `cpp/src/cuda_combustion.h` / `.cu` — `e_deposit_drop_sum` out-param +
  device-side int64 atomic slot; the gas-branch deposit chain rewritten onto
  the wide helper.
- `cpp/src/cuda_temperature.h` / `.cu` — `C_DEP_DROP` slot (6→7),
  `TEMPERATURE_ENERGY_SLOTS` 6→7; the Pass-1 gas-branch deposit chain
  rewritten onto the wide helper.

**Config:**
- `config.toml` — `n_floor_heat` 0.05→0.01 + rewritten rationale (two sites);
  new `[physics.eos] n_work_ref = 0.25`.
- `src/simulation/physics_runner.py` — `n_floor_heat` fallback default;
  `n_work_ref` `_ep` bind.

**Tests / tools:**
- `tests/cuda_combustion_check.py` — 3-tuple rail compare (2 call sites).
- `tests/cuda_po2b_check.py` — 3-tuple rail compare (1 call site).
- `tests/cuda_conduction_check.py`, `tests/cuda_thermal_mass_check.py`,
  `tests/cuda_cool_shift_check.py` — `E_COUNTERS` extended to seven.
- `tools/storm_ledger.py` — `comb.e_deposit_drop_sum` /
  `temp.e_deposit_drop_sum` in `counters()`.
- `tools/e2b_floor_reciprocal_probe.py` — new, the reciprocal-precision
  verification instrument (§1.2), committed.
- `docs/e1_p_e2b_asbuilt_2026-08-17.md` — this record.

**Untracked, regenerable:** `_fire_tuning_artifacts/` (fire_tune_loop CSV/PNG
outputs from the PRE/POST scorecard runs).

**Code commit:** `d8af9d2`.
