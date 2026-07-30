# AUDIT — every constant in the fire / thermal model, classified (2026-07-30)

**Worktree:** `breach.worktrees/fire-const-audit`, detached at `8558dbb`
(`o2-continuous-law` line, with `f5e9aa3` / `344f3ed` / `6f57762` / `b340bba` all
ancestors — i.e. the thermal-mass axis, the cool-shift axis and the full-response
split are all IN).

**Why:** four per-material properties have been found masquerading as globals, one
at a time, each costing a build+bench round trip (`thermal_mass`, `cool_shift`,
`fire_T_ext/span`, `fuel_ref`). Erik: *"seems like it would be sensible to try to
find them all in one go."* This enumerates every constant in the fire/heat chain
and classifies each. Read-only investigation; nothing else in this worktree was
touched.

**Method (§8) is the arc's own lesson**: enumerate exhaustively from the thing
itself — every key in the relevant config sections, every member of the relevant
C++ structs, every literal in the relevant TUs — then classify. Nothing here was
found by grepping for what I expected to find.

---

## 1. Executive summary — the ranked `SHOULD-BE-PER-MATERIAL` list

Only **two materials are flammable today**: `wood` (hp 60, ignition 300) and
`furniture` (hp 30, ignition 280). `door`/`door_closed` carry `ignition_temp = 280`
but `flammable = false`, so they never burn (see §7c). That two-material set is what
"evidence of harm" is computed against — a defect that cannot move either of them is
theoretical.

Two derived facts used throughout, both from the branch's own verified analytics
(`fire_tune_loop.py` header; bench report §2.5/§4, verified ±1% at equilibrium):

- sustain condition `k_die/k_grow < a/(1−a)`, with `a = F·o2f·hot`;
  ambient `o2f = (0.21−0.13)/(1.0−0.13) = 0.0920`.
- flame plateau `T*(I) = k_fire_heat · I · 2^(cool_shift − heat_inv_shift)`,
  hence the survival floor `I_crit = fire_T_ext / (k_fire_heat · 2^(cool_shift − heat_inv_shift))`.

Ranked by **(harm today) × (cheapness to fix)**:

| # | constant | harm today | fix cost | one-line verdict |
|---|---|---|---|---|
| 1 | `fuel_ref` | **DEMONSTRATED — furniture reads F = 0.5 at full health** | cheap (load-time recip table) | already being fixed |
| 2 | `fire_T_ext` / `fire_T_span` | **DEMONSTRATED — both flammables snap out at 350; at the planned 250 the crate starts the race 40 % colder than wood** | cheap (derive as `ignition_temp − Δ`) | the coupling, not the number, is the defect |
| 3 | `k_fire_heat` | none at shipped config; **decisive the moment furniture's `cool_shift` moves off 5** (step 2 of the planned tuning) | very cheap (Python-side per-source scalar) | the GAIN-side twin `cool_shift` never got |
| 4 | `ignition_seed` | none today; couples to (3) — it must clear a now-per-material `I_crit` | cheap (integer column) | should be derived, not set independently |
| 5 | `k_grow` / `k_die` | none (both fuels are cellulosic) | cheap (multiply only) | flame growth rate is a fuel property |
| 6 | `wall_damage` | none | cheap | mass-loss rate is a fuel property; today it rides `hp` |
| 7 | `smoke_emission` | none | cheap | soot yield varies ~10× by fuel in reality |
| 8 | `soot_yield` | none | **NOT cheap** — needs per-claimant soot accounting | see §3 note |
| 9 | `fuel_per_o2`, `burn_rate`, `o2_frac_ext` | none | cheap, but **these are literature anchors** | per-fuel in principle; changing them ≠ turning a dial |
| 10 | `T_FLAME_MAX` / `temp_gain_scale` | none today | cheap, but **blocked on Q2** | adiabatic flame temperature IS a fuel property |

### 1.1 `fuel_ref = 60` — the one with a victim on the shipped build

`F = clamp01(wall_hp / fuel_ref)`, `fuel_ref = 60` **is wood's hp**.

| material | hp | F at full health | `a = F·o2f` (hot=1) | sustain ceiling `a/(1−a)` |
|---|---|---|---|---|
| wood | 60 | 1.00 | 0.0920 | 0.1013 |
| furniture | 30 | **0.50** | 0.0460 | **0.0482** |

A pristine crate's sustain ceiling on `k_die/k_grow` is **2.1× stricter** than wood's,
purely because the normaliser is another material's hp. At Erik's own chosen ratio
(0.080, `fire_tune_loop.py` §3): wood `I_eq = 0.210` ✔, furniture `I_eq = −0.66`
→ **cannot sustain at any intensity**. The tune loop's own caveat block already
records this. Fix = per-material normaliser (== that material's `hp`).
**Determinism:** this is the one candidate carrying a divide, but it needs no runtime
divide — bake `make_recip(hp[mat])` into a per-material `int64` table at load and
index it by the material grid (or hand the kernel a per-tile recip plane). No libm,
no per-cell division.

### 1.2 `fire_T_ext = 350` / `fire_T_span = 150` — global floor under a per-material threshold

`hot = clamp01((T − fire_T_ext)/fire_T_span)`, while `ignition_temp` is already
per-material (wood 300, furniture 280).

- **Shipped today:** `fire_T_ext (350) > ignition_temp` for *both* flammables →
  a freshly ignited tile has `hot = 0` → pure death term → snap-out. This is
  `fire_tuning_plan` §9.2's cold-start gap, still unfixed in config.
- **At the planned `fire_T_ext = 250, span = 100`:** wood ignites at 300 →
  `hot = 0.50`; furniture ignites at 280 → `hot = 0.30`. The crate begins its
  cold-start race with **40 % less growth drive than wood**, for no physical
  reason — the flame's sustain floor is a property of the *fuel*, and each fuel's
  own `ignition_temp` is right there in the table.
- **Blast radius:** any future fuel with `ignition_temp < fire_T_ext` (paper, oil,
  fuel-soaked rag) would ignite and die on the same tick, silently.

**Recommended shape (mirrors the cool-shift vacuum-offset precedent):** keep ONE new
global `ignition_to_ext_delta` and derive `fire_T_ext[mat] = ignition_temp[mat] − Δ`.
That gives every material the right floor with zero new per-material dials, and it
makes the invariant `fire_T_ext < ignition_temp` structural instead of a comment.
`fire_T_span` can stay global or ride the same trick.
**Determinism:** the `− fire_T_ext` is a per-tile subtract (free); only `1/fire_T_span`
is a divide, and it is a load-time reciprocal exactly as today (per-material table if
the span also moves).

### 1.3 `k_fire_heat = 1600` — the gain side that never went per-material

`cool_shift` (loss) and `thermal_mass` (gain conversion) are now per-material.
`k_fire_heat` (gain magnitude) is not. The flame plateau therefore reads
`T* = k_fire_heat · I · 2^(cool_shift − heat_inv_shift)` — two per-material factors
and one global — so `I_crit` is now per-material while the dial that sets it is not.

At the **shipped** config (all `cool_shift = 5`, both flammables `thermal_mass = 8`,
`k_fire_heat = 1600`, `fire_T_ext = 350`): `I_crit = 350/(1600·4) = 0.055` for both.
**No victim today.**

At the **derived tuning point** the session seed hands Erik (`k_fire_heat = 33`,
`materials.furniture.cool_shift = 9`, `fire_T_ext = 250`):

| material | cool_shift | heat_inv_shift | `2^(cs−his)` | `I_crit` |
|---|---|---|---|---|
| furniture | 9 | 3 | 64 | **0.118** |
| wood | 5 | 3 | 4 | **1.89** (>1 → cannot sustain at all) |

and wood's real figure is *worse* than 1.89, because wood has `conductivity = 0.15`
(a second loss channel furniture lacks), which the analytic ignores. So tuning the
crate by moving its `cool_shift` — exactly what step 2 of the tuning order says to do
— makes **wood walls effectively non-flammable**. ⚠ Derived, not measured (§9).

**Fix is nearly free:** `cast_fire_heat` builds each source in Python
(`physics_runner.py:1106`, `src.heat = self.k_fire_heat * intensity_fire`), so a
per-material `k_fire_heat` is one fancy-index lookup on the material grid *outside*
the sim path — no Q16.16 concern at all.

### 1.4 `ignition_seed = 0.1` — a global that must clear a per-material floor

`ignition_seed` is the value `I` is set to at ignition; it must land **above**
`I_crit` or the fire self-collapses at tick 1. Since `I_crit` is per-material through
`cool_shift` and `heat_inv_shift` (§1.3), a single seed cannot be right for two
materials whose `I_crit` differ by 16×. The design seed already asks (Q3) whether
`ignition_seed` should be *derived* from `I_crit` rather than set independently —
this audit says yes, and adds that the derivation is per-material.
Cheap: a quantized integer column, compare/assign only, no divide.

### 1.5 The rest, briefly

`k_grow`/`k_die` (flame growth rate), `wall_damage` (mass-loss rate),
`smoke_emission` (soot yield), `range_base`/`range_per_intensity` (flame reach) are
all genuinely per-fuel physical properties held global. **None has a victim today**
because both flammable materials are cellulosic and wood-like — they are correctness
debt that becomes real the first time a non-wood fuel is added (foam bunk, fuel
drum, plastic crate). All are multiply-only → a per-material quantized table costs
nothing in the sim path.

`fuel_per_o2 = 0.7`, `burn_rate = 0.02`, `o2_frac_ext = 0.13` differ by fuel **in
principle** (stoichiometry, surface reactivity, extinction limit) — but each is a
**literature anchor**, not a dial: making them per-material means finding per-fuel
literature values, not turning a knob. Flagged, not recommended for this batch.

**`H_fuel` is the one that is genuinely, physically global** — see §3.

---

## 2. Full table — `[physics.fire]` (config.toml 115–248)

Values as shipped. `→` in *reason* = "what real physical property differs".

| name | location | value | means | verdict | reason / evidence of harm / blast radius |
|---|---|---|---|---|---|
| `ignition_seed` | config 116; `combat.py:559` | 0.1 | `I` a tile is set to at ignition | **SHOULD-BE-PER-MATERIAL** | → the intensity at which a fuel's flame becomes self-sustaining. Must exceed a now-per-material `I_crit` (§1.4). Harm: none at shipped config (`I_crit` 0.055 both), decisive at the derived tuning point. Blast: wood + furniture; integer column, no divide. |
| `o2_threshold` | config 121 | 0.01 | RETIRED absolute-N_O2 ignition gate | GLOBAL-CORRECT (dead) | tombstoned by the continuous-O2 law; **not read** (§7c). |
| `k_grow` | config 145; `fire_simulation.h:50` | 4.0 | logistic growth gain 1/s | SHOULD-BE-PER-MATERIAL | → flame spread/growth rate on the fuel surface. Harm 0 today (both fuels cellulosic). Blast: multiply-only table. |
| `k_die` | config 146; `.h:51` | 2.0 | decay rate when starved/cold | SHOULD-BE-PER-MATERIAL | as `k_grow`; the RATIO sets `I_eq`, which should differ per fuel. Harm 0 today. |
| `fire_T_ext` | config 147; `.h:52`; `.cpp:198` | 350.0 | flame-sustain floor (game ΔT) | **SHOULD-BE-PER-MATERIAL** | → the fuel's ember/flame sustain temperature, physically tied to its pyrolysis point. **Harm: shipped 350 > both `ignition_temp`s → `hot = 0` at ignition → snap-out; at 250 the crate gets `hot` 0.30 vs wood's 0.50.** Blast: derive from `ignition_temp`; subtract is free. |
| `fire_T_span` | config 148; `.h:53`; `.cpp:111` | 150.0 | width of the `hot` ramp | SHOULD-BE-PER-MATERIAL (with `fire_T_ext`) | same argument; carries a divide → per-material load-time reciprocal table, no runtime divide. |
| `fuel_ref` | config 149; `.h:54`; `.cpp:110,159` | 60.0 | `F = clamp01(wall_hp/fuel_ref)` | **SHOULD-BE-PER-MATERIAL** | → the fuel load of a full tile of THAT material. **Harm: = wood's hp, so furniture (hp 30) reads F = 0.5 pristine and cannot sustain at Erik's ratio (§1.1).** Blast: wood unchanged, furniture doubles its `avail`; recip table. |
| `o2_frac_ext` | config 198; `.h:66`; `combustion.h:139` | 0.13 | flame-extinction O₂ mole fraction | **ANCHORED** (Peatross–Beyler 1997) — per-fuel in principle | limits differ by fuel (~13–16 vol-%); changing it changes a literature anchor. Harm 0 today. Must stay bit-identical between the two O₂ laws. |
| `o2_frac_full` | config 199; `.h:76` | 1.0 | full-response reference (pure O₂) | **ANCHORED** (pure-O₂ reference) | a physical reference, deliberately NOT map-overridden (`b340bba`). GLOBAL-CORRECT. |
| `o2_frac_amb` | config 200; `.h:81` | 0.21 | what the ambient atmosphere IS | GLOBAL-CORRECT (per-MAP, not per-material) | overridden per level by `[ambient] o2_frac`. **No longer read by either O₂ law** (§7c). |
| `P_min` / `P_full` | config 201–202; `.h:88,90` | 0.01 / 0.03 | RETIRED smoothstep edges | GLOBAL-CORRECT (dead) | tombstoned; **not read**. Note config (0.01/0.03) and C++ defaults (0.60/1.00) disagree — harmless only because dead (§7b). |
| `I_min` | config 203; `.h:91`; `.cpp:104,235` | 0.02 | snap-to-zero extinguish floor | GLOBAL-CORRECT | a numerical discretisation floor, not a material property. Interacts with the hard-coded 0.001 early-out (§7b). |
| `k_wind_fan` | config 204; `.h:97` | 0.5 | `1 + k_wind_fan·W` growth fan | SUSPECT-NEEDS-THOUGHT | → how exposed the fuel's surface is to convection. Weak per-material case; scale vs the live wind field is the real open question (config's own note). |
| `k_wind_strip` | config 205; `.h:98` | 0.5 | wind blow-out of marginal fires | SUSPECT-NEEDS-THOUGHT | EOS ruling A2 already earmarked this for replacement by a proper convective term; the tune block sets it to 0.0. Don't make it per-material before that ruling. |
| `fire_pressure_gain` | config 206; `.h:101`; `.cpp:105,280` | 0.15 | plume→T shim gain (1/s) | SUSPECT-NEEDS-THOUGHT | part of the Q2 shim (with `temp_gain_scale`/`T_FLAME_MAX`). If Q2 rules "flame-temperature drive", the trio is a FUEL property. See §5. |
| `p_expand_ref` | config 207; `.h:105` | 1.30 | RETIRED plume saturation gate | GLOBAL-CORRECT (dead) | structurally dead pre-`T_FLAME_MAX`; **not read** (§7c). |
| `smoke_emission` | config 208; `.h:138`; `.cpp:106,311` | 0.8 | smoke/s per unit `I` | SHOULD-BE-PER-MATERIAL | → soot yield, which varies ~10× between fuels. Harm 0 today. NB: a **second, parallel** soot source to `combustion.soot_yield` with a different law (§7b). |
| `wall_damage` | config 209; `.h:139`; `.cpp:107,331` | 0.4 | hp/s per unit `I` (flame-scale fuel burn) | SHOULD-BE-PER-MATERIAL | → mass-loss rate per unit area. Today burn duration ≈ `hp/(wall_damage·I)`, so it rides `hp` — plausible but conflated (§7b: `hp` is structure AND fuel). Harm 0 today. |
| `k_fire_heat` | config 241; `physics_runner.py:291,1106` | 1600.0 | TOTAL radiant heat power per unit `I` | **SHOULD-BE-PER-MATERIAL** | → heat release rate of the burning fuel. **Harm: 0 at shipped config; at the derived tuning point wood's `I_crit` ≥ 1.89 vs furniture's 0.118 (§1.3).** Blast: Python-side scalar — free. |
| `fire_ray_count` | config 242; `runner:292` | 8 | fixed rays per burning tile | GLOBAL-CORRECT | a determinism/quality knob (fixed count, fixed angles, no RNG), not physics. |
| `range_base` | config 243; `runner:293,1098` | 2.0 | flame radiative reach floor (tiles) | SUSPECT-NEEDS-THOUGHT | → flame height/reach, weakly per-fuel. Harm 0. Low value. |
| `range_per_intensity` | config 244; `runner:294` | 3.0 | reach growth with `I` | SUSPECT-NEEDS-THOUGHT | as above. |
| `intensity_base` / `intensity_per_intensity` | config 245–246; `runner:295–297` | 0.3 / 0.7 | ray LIGHT weight | GLOBAL-CORRECT (render) | discarded on the heat-only cast (`physics_runner.cast_fire_heat` passes scratch buffers). |
| `color` | config 247 | `[1.0,0.45,0.12]` | render tint | GLOBAL-CORRECT (render-only) | heat is colourless; buffer discarded. |
| `coarse_cluster` | config 248; `raycaster.h:128` | 3 | fire-source clustering grid | GLOBAL-CORRECT (and currently INERT — §7c) | only `update_from_fire` reads it, which nothing calls. |

## 3. Full table — `[physics.combustion]` (config.toml 379–407)

| name | location | value | means | verdict | reason / harm / blast radius |
|---|---|---|---|---|---|
| `burn_rate` | config 391; `combustion.h:136`; `.cpp:103,203` | 0.02 | N_O2/s per burn site at `I=1, o2f=1` | **ANCHORED** (Huggett 1980 + `ceiling_h`) — per-fuel in principle | → fuel surface reactivity. Harm 0 today. Erik: fixed for the re-tune, "never touched again without a real reason". |
| `o2_thresh_burn` | config 392; `.h:155`; `.cpp:104,165` | 0.03 | epsilon skip-floor on an air cell's O₂ | GLOBAL-CORRECT | retired as a *gate*; now a cheap early-out. A numeric floor, not a material property. |
| `H_fuel` | config 395; `.h:160`; `.cpp:106,275` | 4.0 | ΔT (T-scale) per unit N_O2 burned | **GLOBAL-CORRECT — and physically so** | this is the ONE fire constant with a literature reason to be global: Huggett's principle, ~13.1 MJ per kg O₂ consumed, holds to ±5 % across nearly all organic fuels. Heat released *per unit oxygen* is fuel-independent even though heat per unit fuel is not. **Do not make this per-material.** |
| `soot_yield` | config 396; `.h:161`; `.cpp:105,266` | 0.3 | fraction of consumed O₂ → smoke | SHOULD-BE-PER-MATERIAL (in principle) — **but NOT cheap** | → soot yield is strongly fuel-dependent in reality. **Cost caveat:** it is applied at the AIR cell `j` to the *aggregate* burn from up to 4 claimants of possibly different materials (`combustion.cpp:266`). Making it per-material means splitting the soot deposit per claimant — a structural change to the order-free gather, not a table lookup. Do not batch this with the cheap ones. |
| `fuel_per_o2` | config 404; `.h:163`; `.cpp:109,340` | 0.7 | wall_hp paid per unit N_O2 burned | **ANCHORED** (wood stoichiometry) — per-fuel in principle | charged to the SOURCE tile `i` → per-claimant, so per-material IS cheap here (multiply only, index by `material[i]`). Harm 0 today. Changing it = changing a stoichiometry, not a dial. |
| `o2_thresh_breathe` | config 407; `.h:181` | 0.08 | min N_O2 a unit needs to breathe | GLOBAL-CORRECT (unit property, **not wired**) | deliberate "defined at the right layer, consumed later" (§7c). |
| `FUEL_FLOOR` | `combustion.h:176` (C++ only) | 1 (Q16.16 LSB) | smolder never destroys | GLOBAL-CORRECT | Erik's 1-LSB rule, explicitly "not a dial". |

## 4. Full table — `[physics.thermal]` (config.toml 48–104)

| name | location | value | means | verdict | reason / harm / blast radius |
|---|---|---|---|---|---|
| `TEMP_SCALE` | config 49; `fire_simulation.h:144` | 65536 | Q16.16 scale, == HEAT_SCALE | GLOBAL-CORRECT | a number format. |
| `SHIFT_AT_REF` | config 50; `materials.py:380` | 2 | conduction self-rate at `KAPPA_REF` | GLOBAL-CORRECT | calibration of the log-bucket; the per-material axis is `conductivity`. |
| `SHIFT_MIN` | config 51; also `cool_shift_floor` | 2 | rate floor / stability bound | GLOBAL-CORRECT | a numerical stability bound (4 faces × ¼ ≤ 1, discrete max principle). |
| `KAPPA_REF` | config 52 | 50.0 | reference conductivity (hull) | GLOBAL-CORRECT | a *reference point* for the bucket, not a material property; per-material variation lives in `conductivity`. |
| `NO_FACE` | config 53; `temperature_solver.h:133` | 63 | sentinel: κ==0 / grid edge | GLOBAL-CORRECT | a sentinel. |
| `COOL_SHIFT` | config 71; `temperature_solver.h:169` | 5 | interior ambient decay shift | **ALREADY-PER-MATERIAL** (`materials.*.cool_shift`) — and this global is now **largely inert** | keeps two jobs: (a) default for a row omitting the column — **dead in practice, all 8 rows author it**; (b) the vacuum offset `COOL_SHIFT − COOL_SHIFT_VACUUM`. See §7c. |
| `COOL_SHIFT_VACUUM` | config 72; `.h:170` | 3 | space-exposed decay shift | GLOBAL-CORRECT as an **offset** (deliberate ruling) | the 4× space discount is a property of the BOUNDARY, not the material — but note the documented consequence: raising a material's `cool_shift` also slows its vacuum-exposed cooling (`max(SHIFT_MIN, cs − 2)`). |
| `o2_vacuum_thresh` | config 73; `.h:172`; `.cpp:396,439` | 0.3 | atmosphere below which a neighbour = vacuum | GLOBAL-CORRECT | a property of space, not of the wall. |
| `gas_advection_rate` | config 77; `.h:211`; `.cpp:207` | 900.0 | gas-T wind→displacement scale | GLOBAL-CORRECT | a property of the gas/wind coupling; slated to disappear when `wind` becomes a real velocity. |
| `c_v` | config 81; `.h:212`; `combustion.cpp:110` | 1.0 | gas heat capacity for the ΔT deposit | SUSPECT-NEEDS-THOUGHT (per-GAS, not per-material) | `c_v` is a per-species property applied to the whole mixture (O₂/N₂/steam/soot). The per-gas table exists and could carry it. Harm today: unmeasurable (`c_v` has "no real-gas anchor yet"). Low priority. |
| `n_floor_heat` | config 84; `.h:213` | 0.05 | floor on the per-tile N divisor | GLOBAL-CORRECT | a numerical rail, checked against the single-tick criterion in `temperature_solver.h:198-210`. |
| `T_MAX_PHYS` | config 97; `.h:219`; `combustion.h:188` | 16000.0 | counted physical-max T rail | GLOBAL-CORRECT | a rail on the FIELD, deliberately one constant across EOS/thermal/combustion. PROVISIONAL pending Erik's review — but global is right for a rail. |

## 5. C++ constants with **no config key** (the least visible class)

These are shipped values that exist only as C++ struct defaults — `physics_runner.py`
never binds them, `config.toml` never mentions them. Changing them requires a rebuild
and they are invisible to `--set` overrides and to anyone reading config.

| name | location | value | means | verdict | reason / harm |
|---|---|---|---|---|---|
| `temp_gain_scale` | `fire_simulation.h:115`; `.cpp:265,286` | 50.0 | plume shim: `ΔT = gain · temp_gain_scale` | SUSPECT-NEEDS-THOUGHT + **INVISIBLE** | not in config, not bound. With `fire_pressure_gain = 0.15` the shim deposits `0.3125·I` game/tick **bypassing `heat_inv_shift` entirely**. At `cool_shift = 9` its own steady state is `0.3125·I·2^9 = 160·I` game ⇒ **≈ 34 game units at I = 0.21, ~7 % of the 450 target plateau** (derived from the verified equilibrium relation, not measured — §9). At `cool_shift = 5` it is ~2 game (negligible). Its relative weight therefore GROWS as `k_fire_heat` is tuned down — exactly the risk the session seed §7 flags. |
| `T_FLAME_MAX` | `fire_simulation.h:135`; `.cpp:266,291` | 2000.0 | plume self-limiter ceiling (game ΔT) | **SHOULD-BE-PER-MATERIAL if Q2 rules "flame-temperature drive"** + **INVISIBLE** | adiabatic flame temperature IS a fuel property. 2000 game = 4293 K at the render mapping (`K = 293 + 2·T`), i.e. **far above the §9.3 target band of 400–500 game** — so the smooth taper effectively never engages at the operating point being tuned, and the shim is a near-linear unlimited deposit. Blocked on design-seed Q2; if it is an energy deposit it must instead convert through `heat_inv_shift`. Carries a divide → per-material recip table. |
| `temp_scale` | `fire_simulation.h:144` | 65536 | Q16.16 scale of `temperature` | GLOBAL-CORRECT (bound at `physics_runner.py:222`) | format; identity fast path when == FP_ONE. |
| `X_N_FLOOR` | `fire_simulation.cpp:131`, `combustion.cpp:127`, `cuda_fire.cu:398`, `cuda_combustion.cu:295`, `combat.py:547` | `quantize(0.01)` = 655 | mole-fraction divide floor | GLOBAL-CORRECT, but **duplicated in 5 sites** | correct value (guards `reciprocal_q16`, kills spurious high X on trace gas), but it MUST stay bit-identical across CPU/CUDA/Python and is written out five times as a literal. §7b. |
| `max_fire_thresh` | `fire_simulation.cpp:76`, `cuda_fire.cu:360` | `quantize(0.001)` | whole-step early-out on `max(fire)` | GLOBAL-CORRECT, **coupled to `I_min`** | if `I_min` were ever set below 0.001 the entire fire step would silently no-op. §7b. |
| `GAS_WSUM_FLOOR_Q` / `GAS_WSUM_EPS_Q` | `temperature_solver.cpp:43–44` | `FP_ONE>>8` / `FP_ONE>>14` | gas-T advection renorm floors | GLOBAL-CORRECT | mirrored from smoke's SLint scheme; numerics. |
| `N_AMB == FP_ONE` | `temperature_solver.cpp:299` (implicit) | 1.0 | absorption-proportional deposit reference | GLOBAL-CORRECT, **implicit** | "no new dial" by construction (the P1 calibration makes ambient N quantize to exactly 1.0). Correct, but a reader cannot see the constant — it is the literal `FP_ONE` doing double duty as a physical reference. §7b. |
| `HEAT_SCALE` | `raycaster.h:31` | 65536 | Q16.16 heat format | GLOBAL-CORRECT | format. |
| `light_cull` / `heat_cull` | `raycaster.h:125–126`; `.cpp:226,264` | 0.01 / 0.01 | ray survival floors | GLOBAL-CORRECT | per-CHANNEL floors on the ray, deliberately split so heat can diverge from light; not a material property (the material axis is `heat_atten`). |
| `cool_shift_floor` | `temperature_solver.h:171` | 2 | clamp on the vacuum offset | GLOBAL-CORRECT | bound from `SHIFT_MIN`; load-bearing (prevents shift 0 = `T -= T`). |
| legacy `update_from_fire` literals | `raycaster.cpp:476,500,504–507` | 0.01, 0.1, 15, 0.8, `heat=1.0`, `jitter=0.05` | old cluster-and-cast light path | **DEAD** | no caller in `src/`, `renderer/`, `tools/` or `tests/` — only a pybind export. Note `src.heat = 1.0f` is set but `cast_source` has no heat buffer, and `jitter = 0.05` would be RNG-driven. Harmless because unreachable; a trap if anyone re-wires it. §7c. |

## 6. `[materials.*]` columns in the fire/thermal chain

| column | value range shipped | means | verdict | notes |
|---|---|---|---|---|
| `hp` | air 0, glass 15, furniture 30, door 40, wood 60, steel 200, hull 300 | structural HP **and** the fuel store | ALREADY-PER-MATERIAL — but **conflated** | `wall_hp` is simultaneously structure (blast/bullets) and fuel (`F = wall_hp/fuel_ref`, `fuel_per_o2` payment). Physically fuel load (MJ/m²) and structural strength are independent. The `fuel_ref` fix (§1.1) makes the conflation *consistent*, not gone. SUSPECT for a later arc. |
| `flammable` | true only for wood, furniture | can ignite at all | ALREADY-PER-MATERIAL | gates every fire path. Doors are `false` despite carrying `ignition_temp = 280` (§7c). |
| `conductivity` | 0.0 (furniture) … 50.0 (hull); air 0.024 | thermal conductivity → face-shift table | ALREADY-PER-MATERIAL | consumed via `face_shift` (`materials.py:414-427` → `temperature_solver.cpp:340-357`). Note furniture κ=0 is deliberate (one clean loss channel). |
| `thermal_mass` | 0 (air), 8 (wood/door/furniture), 16 (glass), 32 (steel/hull) | ρ·c analogue; `T += heat >> log2(tm)` | ALREADY-PER-MATERIAL (fixed `f5e9aa3`) | also derives the `thermal_solid` medium mask. Power-of-two contract enforced at load. |
| `cool_shift` | 5 on every row | ambient-decay shift; e-fold `2^cs/24` s | ALREADY-PER-MATERIAL (fixed `344f3ed`) | every row seeded at the old global → byte-identical on arrival; a dial the moment Erik moves one. Validated to `[SHIFT_MIN, 20]`. |
| `ignition_temp` | wood 300, door/door_closed/furniture 280, else 0 | ignition threshold (game ΔT) | ALREADY-PER-MATERIAL | quantized once at load (`materials.py:288`); read by `combat.py:483` and `combustion.cpp:200`. **`fire_T_ext` should be derived from it (§1.2).** |
| `heat_atten` | air 0.0, glass 0.3, furniture 0.5, wood/door/steel/hull 1.0 | heat-ray occlusion | ALREADY-PER-MATERIAL | `raycaster.cpp:303-304`, with the deliberate source-tile self-occlusion skip. |
| `permeability` | furniture 0.5, else derived | gas/smoke flow | ALREADY-PER-MATERIAL (flow axis) | fire-relevant *indirectly*: furniture's 0.5 makes a crate an open cell and therefore a Pass-A **burn site** whose deposit takes the object path (`combustion.cpp:288-293`). |

## 7. Special flags

### 7a. Per-material (or per-gas) columns that **nothing reads** — the `thermal_mass` failure mode

Verified by enumerating *readers*, not by grepping names: for each column I searched
every consumer in `src/`, `renderer/`, `tools/`, `cpp/src/` for the projected grid or
the table attribute.

- **`[gases.*] flammable`** — `gases.py:96` loads it; `fuel_gas` sets it `true`
  ("the only flammable gas … IGNITES → spawns heat + smoke"). **No production reader
  exists.** The only references are `tests/test_multigas_structure.py:155,190`, which
  assert the table loaded. A flamethrower's fuel vapour cloud cannot be ignited.
  This is exactly the `thermal_mass` shape: a column, a documented intent, and no
  routing.
- **`[gases.*] emits_when_hot`** — `gases.py:97`; `smoke` and `fuel_gas` set it
  `true` ("black-body emission driven by the heat buffer — read M2/M3"). **No
  production reader.** Hot soot does not glow.
- **`[gases.*] glow`** — `gases.py:95`, all rows 0.0. **No production reader.**
- **`[physics.fire] coarse_cluster`** — bound onto the raycaster
  (`physics_runner.py:289`) but only `Raycaster::update_from_fire` consumes it, and
  that function has no caller (§5). Inert.

All four are *documented* as "read later" (M2/M3), so they are honest placeholders
rather than bugs — but they are the same shape, and the fire chain is where they
would bite.

### 7b. Units or reference unclear from the code — a future reader could mis-set these

- **`temperature` units.** The field is "game ΔT above ambient", and the only place
  the mapping to Kelvin is written down is the *renderer* config
  (`[render.blackbody] k_temp_to_kelvin = 2.0`, `kelvin_ambient = 293.0`, i.e.
  `K = 293 + 2·T`). Every sim-side threshold (`ignition_temp`, `fire_T_ext`,
  `T_FLAME_MAX`, `T_MAX_PHYS`) is authored in that unit with no anchor in the sim
  config. A reader who assumes Kelvin (or Celsius) will be wrong by 2×.
- **`H_fuel = 4.0` "T-scale ΔT per unit N_O2"** — the units chain
  (N_O2 counts → ΔT before the `/(N·c_v)` or `>>heat_inv_shift` conversion) is
  described only in prose. There is no stated physical anchor despite the Huggett
  citation sitting one comment block away.
- **`c_v = 1.0` "neutral scale, no real-gas anchor yet"** — explicitly unanchored by
  its own comment. Anything derived from it inherits that.
- **`X_N_FLOOR = quantize(0.01)`** appears as a bare literal in **five** places
  (CPU fire, CPU combustion, CUDA fire, CUDA combustion, Python ignition). It must
  be bit-identical in all five; nothing enforces that but comments.
- **`max_fire_thresh = quantize(0.001)`** vs **`I_min = 0.02`** — an undocumented
  coupling. Setting `I_min` below 0.001 makes the entire fire step early-out.
- **`N_AMB == FP_ONE`** (`temperature_solver.cpp:299`) — a *physical reference*
  (ambient bulk density) encoded as the fixed-point literal `FP_ONE`, with the
  equality holding only because of the P1 calibration. Correct and documented in the
  comment, but invisible as a constant.
- **`P_min`/`P_full`: config says 0.01/0.03, C++ defaults say 0.60/1.00.** Harmless
  only because both are dead. If anything ever re-reads them, the two sources of
  truth disagree.
- **Two independent soot sources.** `[physics.fire] smoke_emission = 0.8` (per
  second per `I`, into neighbour air, `fire_simulation.cpp:311`) and
  `[physics.combustion] soot_yield = 0.3` (fraction of consumed O₂,
  `combustion.cpp:266`) both make smoke, by different laws, into different planes,
  with no stated relationship. Tuning one does not obviously interact with the other,
  but physically they are the same phenomenon.
- **`temp_gain_scale` / `T_FLAME_MAX` are not in `config.toml` at all** (§5) — a
  reader auditing the fire model from config will not see the plume shim's two most
  important numbers.

### 7c. Silently inert under current settings

The `cool_shift` trap generalised — constants that are still bound, still documented,
and no longer do anything:

| constant | why inert | risk |
|---|---|---|
| `[physics.thermal] COOL_SHIFT` as a *decay* dial | **all 8 material rows author `cool_shift` explicitly**, so the "default for a row that omits it" job is dead too — only the vacuum-offset job survives | ★ known trap; `--set physics.thermal.COOL_SHIFT=N` moves nothing |
| `[physics.fire] o2_frac_amb` | no longer read by either O₂ law since `b340bba`; kept as the ambient record | tuning it expecting a fire change does nothing |
| `[physics.fire] o2_threshold` | retired by the continuous-O2 law | tombstoned, documented |
| `[physics.fire] P_min` / `P_full` | retired smoothstep edges | tombstoned, documented |
| `[physics.fire] p_expand_ref` | retired plume gate (was structurally dead: it read `atmosphere` at a SOLID tile, which the EOS force-zeroes) | tombstoned, documented |
| `[physics.combustion] o2_thresh_burn` | retired as the burn GATE; survives only as an epsilon skip-floor | a reader may still think it gates burning |
| `[physics.combustion] o2_thresh_breathe` | defined, nothing reads it (deliberate: later arc) | documented non-goal |
| `[physics.fire] coarse_cluster` | its only consumer has no caller | inert |
| `Raycaster::update_from_fire` (whole function) | no caller anywhere outside the pybind export | dead code carrying an RNG jitter |
| `door` / `door_closed` `ignition_temp = 280` | `flammable = false` gates every fire path first | **a real inconsistency**: canon ch.02 illustrates doors as flammable; config keeps them non-flammable "to preserve existing physics behaviour". The threshold is authored and unreachable. |
| `[gases.*] flammable` / `emits_when_hot` / `glow` | no readers (§7a) | fuel vapour cannot ignite |
| `T_FLAME_MAX` taper, at the tuning target | 2000 game ≫ the 400–500 game target band, so the smooth self-limiter never engages | the shim behaves as an unlimited linear deposit at the operating point being tuned |

## 8. Method

1. Enumerated **every key** in `[physics.fire]`, `[physics.thermal]`,
   `[physics.combustion]`, and the fire-relevant part of `[physics.eos]` by reading
   `config.toml` 16–410 line by line, then cross-checked the list against
   `grep -n "^<key>"` so no key was missed by eye.
2. Enumerated **every member** of `FireParams`, `CombustionSolver`,
   `TemperatureSolver` and the fire-relevant part of `Raycaster` by reading the
   headers end to end — this is what surfaced `temp_gain_scale` / `T_FLAME_MAX`,
   which have no config key and would never appear in a config-driven sweep.
3. Read `fire_simulation.cpp`, `combustion.cpp`, `temperature_solver.cpp` in full and
   the heat path of `raycaster.cpp`, collecting **bare literals** (`quantize(0.001)`,
   `quantize(0.01)`, `FP_ONE>>8`, `FUEL_FLOOR`, the `update_from_fire` block).
4. Enumerated **every column** of `MaterialTable` and `GasTable` from
   `_SCALAR_COLUMNS` plus the optional-column blocks, then for each one searched for
   its **readers** (the projected grid or the table attribute) across
   `src/`, `renderer/`, `tools/`, `cpp/src/`, `tests/` — the ruling §5 rule: verify
   ownership by enumerating readers/writers, never by grepping a name near a keyword.
   That is what produced §7a.
5. Checked the binding layer (`physics_runner.py`) for each config key, which is what
   distinguishes "shipped value" from "C++ default nobody overrides".

## 9. What I could **not** determine — marked, not guessed

- **Nothing here was measured.** No build was run, no bench executed (read-only
  audit). Every number in §1.3 and §5 is **derived** from relations this branch has
  already verified (`T*(I)` to ±1 % at equilibrium; the `I_eq` fixed point), applied
  to config values I read. The direction of each conclusion is robust; exact
  magnitudes are not measurements.
- **Wood's true `I_crit`** at the derived tuning point. The `T*(I)` analytic was
  established for furniture, whose *only* loss channel is `cool_shift`
  (`conductivity = 0`). Wood conducts (κ = 0.15, wood–wood face shift 8), so it has a
  second loss channel the analytic omits — its real `I_crit` is **higher** than the
  1.89 I computed, i.e. the finding is conservative in the direction that matters,
  but the number itself is a lower bound, not a value.
- **Whether the plume shim should be per-material at all** — that is design-seed Q2
  (energy deposit vs flame-temperature drive), explicitly open. I classified
  `T_FLAME_MAX` conditionally rather than asserting a verdict.
- **Whether `[gases.*] flammable` / `emits_when_hot` are "not yet wired" or "wired
  somewhere I did not search."** I searched `src/`, `renderer/`, `tools/`, `cpp/src/`
  and `tests/` for the table attributes and found only test assertions; I did not
  audit shader code (`shaders/`) for the `emits_when_hot` path, and the B2 render arc
  may consume the black-body channel by a different route.
- **Weapon/payload-side fire seeding** (`[payloads.incendiary_splash]
  ignite_intensity = 0.5`, `ignite_radius = 2.5`) seeds `I` directly, bypassing
  `ignition_seed` and therefore bypassing the `I_crit` question. I noted it but did
  not audit the weapons tables — out of the stated scope.
- **CUDA twins were checked for constant *divergence* only** (the literals in
  `cuda_fire.cu` / `cuda_combustion.cu` match their CPU counterparts, and the
  device kernels take every dial as a parameter). I did not re-derive the CUDA
  numerics.
