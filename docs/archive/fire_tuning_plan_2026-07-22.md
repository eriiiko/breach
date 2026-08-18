# Fire & Heat — tuning procedure and parameter queue

**Started 2026-07-22 (Erik + Claude). LIVING working doc, multi-session.** One
parameter (or one tight group) at a time. Companion to the Fire & Heat Beauty
arc; the B2 *render* look ships on its own branch — this doc is the **sim-side
fire/heat physics tuning**.

Reference for the current model + values: `config.toml` `[physics.fire]`,
`[physics.combustion]`, `[physics.thermal]`, the `[materials.*]` table; the
fire model in `cpp/src/fire_simulation.*` + `src/simulation/fire_fixed.py`; the
render Kelvin mapping in `renderer/blackbody.py`.

---

## 0. How we work this (agreed 2026-07-22)

- **One param (or tight group) per session.** Understand → hypothesize → tune
  live on the burn-bench → Erik picks a value → commit to `config.toml` →
  **deliberately re-baseline the affected goldens with written rationale** →
  HUMAN-TEST.
- **This is sim-side.** Unlike the B2 render work, these dials move
  digests/goldens. Every accepted change is a deliberate golden re-baseline
  with rationale (project iron rule) + Erik plays before merge.
- **Physically-anchored params are tuned LAST** (Erik's rule): a value that is
  already real physics (stoichiometry, the gas law, air composition) is not a
  free dial — we only touch it if forced, and with a physical justification.
- **Order approved by Erik 2026-07-22** (the §4 queue below). Within the
  single-flame phase we start by *understanding* `I` and tuning `k_grow`,
  per Erik's request.

---

## 1. FOUNDATION — units: KEEP game units  ⟶ **DECISION #1 (Erik, 2026-07-22)**

**Locked:** we KEEP the existing **game-unit** temperature scale — ambient = 0,
cleverly thought out, and the ignition temps are good as-is. We do NOT migrate
the field to Kelvin. Kelvin/Celsius are for OUR reasoning during tuning, via a
conversion helper; the sim field stays game units.

**The scale (render blackbody mapping, unchanged, `renderer/blackbody.py:199`):**
```
Kelvin  = 293 + 2·T_game          (k_temp_to_kelvin = 2, kelvin_ambient = 293)
T_game  = (Kelvin − 293) / 2
Celsius = Kelvin − 273.15          (0°C = 273.15 K)
T_game  = (Celsius − 19.85) / 2    (ambient 20°C → T_game ≈ 0 ✓)
```
Reference points: **ambient 0** (20°C) · **furniture ignition 280** (853 K,
580°C) · **wood ignition 300** (893 K, 620°C) · **fire_T_ext 350** (993 K,
720°C). Physically sensible (ignition few-hundred °C, sustain ~700°C).

**Tuning helper (this session):** a tiny K/°C ↔ game-units converter so we can
author experiments in physical temperatures. Formulas above; a `tools/`
one-liner if we want it live.

*(Superseded: a brief "absolute Kelvin field" call made before the scale was
clarified — reverted 2026-07-22. Game units stay.)*

---

## 2. The model — how a fire works today

### 2.1 Two distinct fields
- **`temperature` (T)** — heat state per tile, **game units** (ambient = 0;
  ×2+293 → Kelvin, §1). *No setpoint*: fire deposits heat *energy* each tick; T
  is the running balance of deposit − conduction − ambient cooling, so **T rises
  over time** while burning (toward an equilibrium, capped by the `T_MAX_PHYS`
  rail) and falls back toward 0 (ambient) when the fire dies.
- **`I` = fire intensity ∈ [0,1]** (`fire` field) — *how vigorously this tile
  burns* (0 unlit, 1 fully ablaze). A dynamical variable, **not** temperature,
  coupled to T through the `hot` gate below.

### 2.2 Ignition (a new tile catches)
Cellular spread is **deleted**. The only path is **heat → temperature →
ignition**: a flammable tile whose `T ≥ ignition_temp` *and* has local O₂ gets
seeded `fire = max(fire, ignition_seed = 0.1)`.

### 2.3 The intensity logistic (life/death of a lit tile)
Per burning tile, per tick:
```
F     = clamp01(wall_hp / fuel_ref)                 # fuel remaining
o2    = smoothstep(P_min, P_full, local_O2)         # oxygen gate
hot   = clamp01((T − fire_T_ext) / fire_T_span)     # heat gate
avail = F · o2
grow  = k_grow · avail · hot · I·(1−I) · (1 + k_wind_fan·W)
die   = k_die  · (1 − avail·hot) · I  +  k_wind_strip·W·(1−I)·I
I    += dt·(grow − die);   clamp [0,1];   if I < I_min → 0
```
- `I·(1−I)` = logistic S-curve: fastest growth mid-way, self-limits at 1.
- **`k_grow`** = growth rate (1/s); **`k_die`** = decay rate (1/s).
- **O₂/fuel/heat gate growth and drive death** via `avail·hot`: well-fed
  (≈1) → grows to full; starved (→0) → dies and snaps out.
- Coupling loop: fuel+O₂+heat → `I` grows → `I` emits heat/smoke/damage →
  heat raises T → `hot` feeds back into `I`.
- **Note the ordering wart:** today `fire_T_ext = 350` (extinction) >
  `ignition_temp = 300` (wood) — a tile ignites at 300 but can't *grow* until
  350. Revisit in the fire-life phase (§4.2d).

### 2.4 Heat radiation & spread  *(symptom 1: "reaches half the room")*
Each burning tile casts an 8-ray heat source into `gmap.heat` at step start:
`heat = k_fire_heat · I`, reach `range_base + range_per_intensity · I` tiles.
That heat conducts into neighbours' `temperature`; neighbours crossing
`ignition_temp` ignite. With `k_fire_heat = 1600` and reach `2 + 3·I ≈ 5`
tiles, one intense crate flashes everything within ~5 tiles almost at once.

### 2.5 O₂ & fuel consumption
Separate combustion pass (once/tick, on real O₂): burns `burn_rate = 1.0` O₂
per second per burning site → heat (`H_fuel = 4.0` per O₂), soot
(`soot_yield = 0.3`), inert N₂ (rest; N_total conserved). Fuel = `wall_hp`
(wood 60, crate 30), drained by `wall_damage = 0.4·I`/s (destructive burn-out)
and `fuel_per_o2 = 0.7` per O₂ (smolder, never destroys).

### 2.6 Pressure & burst coupling  *(symptom 2: "room bursts before it chokes")*
Fire heat → hot gas → `P = C·N·T` rises (+ `fire_pressure_gain` plume push).
When local P exceeds a wall's `burst_threshold` (only **2.0** for wood/crate)
the wall pops — and that beats O₂-starvation (~265 ticks measured), so the room
vents before it chokes. Fixing this = balance the heat→pressure chain
(`H_fuel`, `k_fire_heat`, `fire_pressure_gain`) vs `burst_threshold` so the
O₂-choke wins.

---

## 3. Materials & ignition temps

**8 material rows.** Only flammable ones have a meaningful `ignition_temp`.
Ignition temps **KEPT as-is** (Decision #1 — game units, they're good); physical
equivalent shown for reference.

| material | flammable | hp (fuel) | ignition_temp (game) | ≈ physical |
|---|---|---|---|---|
| air | no | 0 | 0 | — |
| hull | no | 300 | 0 | — |
| **wood** (wall) | **yes** | 60 | 300 | 893 K / 620°C |
| door | no* | 40 | 280 | 853 K / 580°C |
| steel | no | 200 | 0 | — |
| glass | no | 15 | 0 | — |
| **furniture** (crate) | **yes** | 30 | 280 | 853 K / 580°C |
| door_closed | no* | 40 | 280 | 853 K / 580°C |

Open question: **should doors be flammable?** (Today hard-set false to preserve
behavior — `[materials.door]` note.) Deferred decision.

---

## 4. The parameter queue (tackle in this order — approved 2026-07-22)

Status key: ☐ todo · ◐ in progress · ☑ locked (committed + rationale).

**Phase 0 — foundation**
- ☑ **Decision #1**: keep game units (§1); no Kelvin migration. Optional
  K/°C ↔ game-units helper for authoring experiments.
- ☐ **Burn-bench level** (§6): isolated single crate, open ventilated room.

**Phase 1 — the single flame** (REVISED 2026-07-23 — the ramp is COUPLED, see §5.3):
- 0. ☐ **Unblock: `k_wind_strip → 0`** — drop the plume self-blow-out (wind
  blow-out moves to convective cooling, Phase 3). Prereq for ANY slow flame; the
  `k_wind_strip` design fork pulled forward (Erik's convective-cooling idea).
- a. ☐ **Fuel timescale** — `fuel_per_o2` / `burn_rate` / `wall_damage` down so
  the crate LIVES ~5–10 min (the burnout target). Sets the clock the ramp fits in.
- b. ☐ **`k_grow` / `k_die`** (keep 2:1) — ramp to peak in ~2–3 min; isolatable
  once (0)+(a) are set.
- c. ☐ fine-tune fuel for exact burnout; then `k_fire_heat`/`H_fuel` vs
  `COOL_SHIFT` (heat), `o2_thresh_burn` (O₂ draw), the `fire_T_ext`>ignition
  wart (§2.3).

**Phase 2 — spread** *(fixes symptom 1)*
- ☐ `range_base`, `range_per_intensity`, `k_fire_heat` — fire *creeps* believably, no instant flashover. Bench: spaced crate pair.

**Phase 3 — environment coupling** *(fixes symptom 2)*
- ☐ `burst_threshold` vs `H_fuel`/`fire_pressure_gain`/`k_fire_heat` — O₂-choke beats the burst in a sealed room.
- ☐ `k_wind_fan`, `k_wind_strip` — wind fanning/blow-out vs the live wind scale.
- ☐ `P_min`, `P_full` — the O₂ gate (currently provisional).

**Phase 4 — output / handover**
- ☐ `smoke_emission` — soot produced (feeds the B2 render medium).

**Phase 5 — physically anchored (LAST, only if forced)**
- ☐ `soot_yield` (0.3), `fuel_per_o2` (0.7 wood stoich), EOS
  (`adiabatic_index` 1.4, `t_amb_k`, `C`), air composition (0.21/0.79).

---

## 5. Calibration experiments

### 5.1 `k_fire_heat` — the room-flashover clock (Erik's leading star)
Informal target: **a single big fire in the middle of a room should, after
~5–10 minutes, accumulate enough room temperature that everything ignites
("overignites").** So we set `k_fire_heat` (given the cooling rate `COOL_SHIFT`
and a reference room size/ventilation) to hit a 5–10 min time-to-flashover.
Bench: one sustained fire, log mean room temperature vs sim-time, find the
`k_fire_heat` that reaches whole-room ignition in that window. (Couples to
`COOL_SHIFT` and room volume — fix those as the experiment's constants.)

### 5.2 Single-crate ramp & burnout (Phase 1a) — first-principles numbers

Analysis of the intensity logistic for one crate in an open, O₂-saturated room
(`o2 = 1`), assuming it's already hot (`hot = 1`; real ramp adds a thermal
warm-up — confirm on the bench-sim):

Let `a = k_grow·avail·hot`, `d = k_die·(1−avail·hot)`. Then
`dI/dt = I·[(a−d) − a·I]` → **equilibrium `I_eq = 1 − d/a`**, **growth rate
`r = a − d`**.
- **Ratio `k_grow/k_die` sets the max intensity; magnitude sets the speed.**

**Crate** (`hp=30 < fuel_ref=60` → `F=0.5`, so `avail·hot = 0.5`):
`a = 4·0.5 = 2`, `d = 2·0.5 = 1` → **`I_eq = 0.5`** (a lone crate is
fuel-limited to half-ablaze, *never* I=1), `r = 1/s` → ramp `I: 0.1→0.9·I_eq`
in **≈ 3.6 s**. **Wood wall** (`F=1`): `I_eq = 1`, `r = 4/s` → ramp **≈ 1.1 s**.

**So today a crate flashes up in ~seconds — the "all hell breaks loose" feel.**

**Targets (Erik):** ramp low→ablaze **3–5 min**; burnout **5–10 min**.
- To hit a **4-min crate ramp keeping `I_eq=0.5`**: scale both dials down ~65×
  (preserve the 2:1 ratio) → **`k_grow ≈ 0.06`, `k_die ≈ 0.03`**. (Bench-sim
  confirms the exact value with thermal warm-up folded in.)
- **Burnout (analytic):** crate `hp=30` drained by `wall_damage·I` (0.4·0.5≈0.2)
  + combustion `fuel_per_o2·burn_rate` (0.7·1.0=0.7) ≈ **0.9 hp/s → ~33 s** at
  steady burn. Want 5–10 min → drain **~10–18× slower** (Phase 1b:
  `wall_damage`, `fuel_per_o2`/`burn_rate`, `fuel_ref`).

**Two design questions for Erik:**
1. A lone crate maxes at **I=0.5** (fuel-limited). Accept that as "a crate's
   full burn," or make crates reach higher (lower `fuel_ref`, or give crates
   more `hp`)?
2. Confirm the exact targets: ramp = 3 / 4 / 5 min? burnout = 5 / 7 / 10 min?

**Next:** build the burn-bench and run the *real engine* single-crate sim to
confirm these numbers (thermal warm-up + the C++ combustion coupling), then tune
`k_grow`/`k_die` live to the agreed ramp target.

### 5.3 The harness + 2026-07-23 findings (real-engine numbers)

**The instrument:** `tools/fire_timing_harness.py` (fire-tuning worktree).
Headless, deterministic, planetside (O₂ held ~0.21 near the ambient ring), one
furniture crate ignited at the seed. **How we tune — via `--set`, config.toml
untouched until a value is blessed:**
- default / `--wind 0` → **NATURAL wind** (the fire's own plume + convection —
  the realistic still-air baseline; we do NOT force wind to 0).
- `--wind W` → forced uniform constant wind (windy scenario only).
- `--wind-sweep` → the W → wind-speed(m/s) → burnout mapping.
- `--k-grow V`, `--k-die V`, repeatable `--set key=value` (bare → `[physics.fire]`;
  dotted → e.g. `--set physics.combustion.fuel_per_o2=0.05`) → override any dial
  live without editing config.

**Still-air baseline (natural wind, current dials):** peak I ≈ 0.43, steady
T ≈ 5026, **burnout ≈ 39 s**. Confirms "fire too fast" + the §5.2 analytic.
**Wind calibration:** drift(m/s) ≈ 0.8·W; frisk vind ≈ 10 m/s ⟶ W ≈ 12–13.

**★ KEY FINDING — the ramp is COUPLED, not isolatable.** Scaling (k_grow,k_die)
down alone STALLS the fire at the seed. Cause: a burning tile makes its OWN
plume-wind (~1–2) in still air, and the `k_wind_strip` blow-out term reads that
self-wind and snuffs a slow flame (self-blow-out ~0.067/s ≫ grow ~0.004/s at
k_grow 0.08); at fast k_grow=4 the fire outruns it, a slow one can't. PLUS fuel
is gone in ~40 s regardless. **Proven fix:** `k_wind_strip=0` + slowed fuel →
the same slow (0.08,0.04) pair ramps properly. So `k_grow` is tunable only AFTER
the plume self-blow-out is removed and the fuel timescale is set — this is the
`k_wind_strip` design fork (Erik's convective-cooling idea), pulled forward.

**★ Two model quirks the harness exposed** (parked, §7):
1. A burning tile THINS its own local O₂ (hot gas expands, P=C·N·T) → a crate
   deep in a room self-starves even planetside; it must sit near the ambient
   ring to stay O₂-fed. "O₂-rich" needs airflow, not just volume.
2. **Zombie smolder:** `wall_hp` runs NEGATIVE and `apply_temperature_ignition`
   re-seeds I=0.1 every tick while T>ignition_temp, so a crate never truly goes
   out — likely a big driver of "fire never stops." Fix candidate: re-ignition
   must require fuel (`wall_hp>0`) / a burning tile dies below ignition_temp.

---

## 6. The burn-bench scenario
A minimal diagnostic level (generator like `gen_fire_studio`, but bare):
- One large, **well-ventilated** open room (plenty of O₂, no pressure trap) so
  a single flame burns in isolation — read *how hot, how fast, how long*.
- A **spaced pair** of crates (a few tiles apart) for deliberate spread tuning.
- Optional sealed sub-room later for the O₂-choke / burst phase.
No beacon/marines/theatrics — this is instrumentation, not a showcase.

---

## 7. Handoff to Fable — O₂ / boundary-conditions / combustion redesign

The still-air single-crate **ramp + burnout are tuned + LOCKED** (values below).
The **O₂ behavior is NOT** — Erik is taking it to a **Fable design session first**,
because it needs a MODEL decision before `burn_rate` can be tuned, and it may
require touching the physics engine. Erik will ALSO test independently in the
**level editor** (a hand-made planetside level) — he does not yet trust the BC.
Answer these, roughly in order:

**Q1 — Intensity ↔ O₂: make it CONTINUOUS.** Today the O₂ dependence is a
smoothstep GATE (`o2 = smoothstep(P_min=0.01, P_full=0.03, O₂)`) — full burn
above 0.03, off below 0.01, a near-threshold. Erik wants a **proportional law**:
half O₂ → burn half as much, 10% O₂ → 10% as much. Redesign the O₂→burn coupling
(the gate and/or the combustion draw) to be continuous.

**Q2 — Planetside BC: does O₂ replenish, and should it come "from above"?**
Verified: the ambient RING is clamped to the infinite reservoir (0.21) every
substep (`cpp/src/bulk_transport.cpp` "AMBIENT ring reset"); the INTERIOR is
refilled ONLY by diffusion/advection from the ring — an **EDGE reservoir, not a
volumetric source**. Measured: one crate zeroes a ~20 m open-field room's O₂ in
~5 min (far-O₂ 0.21→0.05 @1min → 0.019 @2min → 0 @5min), while the flame's own
gate stays saturated (ring-adjacent + thermal thinning). **Erik's proposal:** an
open planetside field should replenish O₂ **from above** (the sky / 3rd
dimension), which the 2-D edge-ring model misses. Should the ambient BC gain a
volumetric "outdoor" O₂ refill? → the "**can I trust the BC?**" question.

**★ Q2 RESOLVED (2026-07-23, Fable session).** The BC is CORRECT and IDENTICAL
in harness + level editor (both: SPACE ring → `is_ambient`, per-substep N clamp
on ring tiles only — `cpp/src/bulk_transport.cpp` "AMBIENT ring reset"). The
plot's depletion + asymmetry are fully explained, no bug:
1. **Edge-line reservoir vs volumetric consumption** — `burn_rate=1.0` eats
   ~4.8 tile-loads of O₂/s; the 1200-tile bench holds 252 tile-loads → gone in
   ~4 min even with a working ring.
2. **No suction while burning (Erik's hypothesis CONFIRMED)** — combustion
   conserves N_total and fire heat RAISES P=C·N·T, so bulk wind points OUTWARD
   the whole burn; O₂ can only diffuse in against it. Refill starts at death.
3. **The sponge shaped the picture** — the harness inherited the DEFAULT
   u-damping band (width 8, k_max 0.9): all but ~6 mid rows of the 60×20 bench
   were velocity-damped. The one-sided rightward tongue = crate at x=1 (leftward
   flux clamped into the ring, rightward jet down the sponge-free mid channel,
   stalling in the right band).

**DECISION (Erik): Option A — sky exchange, composition-swap variant.** Every
SKY-CONNECTED air tile (flood fill from the ring through open air, rebuilt on
structural change — the sponge-BFS pattern; sealed rooms correctly excluded)
relaxes composition at FIXED local N_total:
`ΔN = λ·(o2_frac·N_tot − N_O2); N_O2 += ΔN, N_inert −= ΔN` per tick, Q16.
Zero pressure footprint → no EOS/wind coupling. λ = vertical-mixing timescale
dial (τ ≈ 30–120 s, tune on the bench). Scope: **O₂/N₂ only at first**; smoke's
sky-λ (upward removal outdoors — realistic, never forces smoke down) DEFERRED
to a B2-adjacent look decision; T needs no term (`COOL_SHIFT` ambient cooling
IS the vertical heat channel). Accepted approximation: a roofed room with an
open door is sky-connected and gets refill (authored roof mask later if it
bites). Option B (buoyant-chimney mass venting → real in-plane suction) = a
possible LATER feel addition, not a prerequisite. CPU + CUDA-resident mirror;
digests move on planetside maps only.

**SPONGE sub-decision (Erik): EMA HIGH-PASS sponge.** Absorber stays, same
width everywhere (shockwave reflections). Chosen fix: damp `(u − ū)` where ū is
a slow per-tile EMA (τ ≈ 5–10 s, power-of-two Q16 shift) — acoustic fronts are
fast and get eaten, steady wind tracks into ū and passes. Needed because WINDY
levels are planned (outdoor + indoor). Runs as its OWN design-gated engine item
(touches the CUDA resident path), NOT blocking fire tuning: Q3 is volumetric
once A lands, forced-wind harness runs bypass the sponge by construction, and
sealed rooms have no band. Bench hygiene NOW: benches keep the authored sponge;
size interiors so the instrumented region sits ≥ sponge_width + a few tiles
from every ring (the 60×20 bench FAILS this vertically).

**ceiling_h — the canonical air column EXISTS and is consistent.** `config.toml`
`[physics.water] ceiling_h = 2.5` m — declared "ONE constant for the air
column" (`physics_runner.py:390`; consumers: W3 water displacement + the water
solver's CFL `h_ref`; the F-key T-paint aid also assumes 2.5 m). Any new
consumer (Q3 realism math, sky-exchange interpretation) reads the SAME key;
promote it out of `[physics.water]` when a third system lands. Q3 preview with
it: tile air = 0.333²·2.5 ≈ 0.28 m³ → ~0.08 kg O₂/tile; `burn_rate=1.0` ≈
0.37 kg O₂/s vs a real ~50–100 kW crate fire's ~0.004–0.008 kg/s (Thornton's
rule, ~13.1 MJ/kg O₂) → **~50–90× too fast; expect burn_rate O(0.01–0.02)**,
subject to gameplay. NEXT: **Q1 — the continuous O₂ law.**

**Q3 — THEN tune O₂ consumption (`burn_rate`).** `burn_rate=1.0` (O₂/s/site) was
NOT changed in the lock (it doesn't gate the ring-adjacent crate). Too fast?
Decide ONLY after Q1+Q2 fix the setup — depletion is a consumption × replenishment
balance.

**★ Q3 MEASURED (2026-07-23, harness, `--set physics.combustion.burn_rate=0.02`,
locked combo otherwise):** the physically-anchored value (~1/50, from
`ceiling_h`-column realism — see the Q2 block) UNRAVELS the blessed flame:
peak I 0.61→**0.81**, steady T 5026→**~15000, hitting the T_MAX_PHYS 16000
rail**, O₂ gate saturated (1.00) throughout, burnout 9.12→**~23 min** (smolder
drain `fuel_per_o2·burn_rate` collapses 0.045→0.0009, leaving only
`wall_damage·I`). Diagnosis: the locked feel was partly O₂-SELF-STARVATION —
the fast draw thinning the crate's own oxygen capped I and T; remove it and
the o2 gate stops doing any work. **Erik's intent stands (burn_rate 0.02,
touch it never again), but the cut must land INSIDE the Q1 session as ONE
re-tune**, not as a lone config edit: the Q1 continuous O₂ law replaces the
gate that self-starvation was feeding, then `k_grow/k_die` (peak) +
`wall_damage` (burnout ~0.083 for ≈7.5 min at Ī≈0.8) re-fit around
burn_rate=0.02, ONE golden rebase + ONE HUMAN-TEST at the end. Config
UNTOUCHED today (blessed values preserved). Side-finding for §5.1: with O₂
saturated, the heat chain RAILS at T_MAX_PHYS — the locked 5026 steady-T was
starvation-limited, not cooling-limited; remember this when tuning
`k_fire_heat` vs `COOL_SHIFT`.

**Q4 — Do we even NEED intensity, or is temperature enough?** (Ask this IF the
redesign touches the engine anyway.) Erik: T looks more physical + would give
nicer dynamic lighting; maybe the fire signal should be T, not a separate I.
Counter-weight: **I is the [0,1] combustion state** that drives heat/smoke/fuel/O₂
and gave the tuned ramp/sustain/blow-out dynamics; **T is the accumulated heat**.
Coupled but distinct — collapse to T alone and any hot tile "burns." Weigh it.

**Q5 — Combustion chemistry check.** Today consumed O₂ → soot (`soot_yield=0.3`)
+ "inert_N2" (0.7), N_total conserved. Real combustion O₂ → CO₂/H₂O (+soot), NOT
N₂ — the sim lumps CO₂/H₂O into the "inert_N2 = nitrogen + burnt products"
catch-all plane. Erik: "when O₂ is consumed it turns to N₂ — is that correct?"
Keep the lump, or track CO₂/H₂O?

**Q6 — Soot yield should be O₂-DEPENDENT (dirty combustion).** Erik: **hot + LOW
O₂ → lots of soot (black smoke); hot + HIGH O₂ → little soot.** Today
`soot_yield=0.3` is fixed. Make it a function of local O₂ (starved = sooty/black,
rich = clean). This is the physical basis of the B2 "dirty-Planck" speckle + the
O₂-choke visual story — same O₂-dependent soot law should drive both.

**★ Q6 RESOLVED (2026-07-24, Fable session + Erik's architectural call):**
design doc `docs/smoke_single_source_design_2026-07-24.md`. Fire smoke gets ONE
source: **source A (the logistic's `smoke_emission·I` scatter) is DELETED**;
the combustion soot channel becomes sole source, its yield now the Q6 law
`y(o2f) = y_clean + (y_starved−y_clean)·(1−o2f)` (starved = sooty, Tewarson),
with soot mass drawn from **O₂ AND fuel** (`k_fuel_soot`; trace-plane deposit,
pressure-neutral — verified: smoke is not in the EOS bulk pair). Emergent:
smoke PEAKS mid-choke, stops at death. Cost: one-time ~10–40× re-gain of all
smoke-density readers (primary seam: `[gases.smoke]` optics). Builds AFTER
`o2-continuous-law` (needs `o2f`), stacked; dials felt at the joint re-tune.
NOTE: Erik's dirty-Planck flame-darkening stays a SEPARATE render experiment
(evaluable only post-re-tune, when fires idle); `o2f` will be available to it.

**Q7 — (render) Fire LIGHT ← temperature, for flicker.** Erik: tie the fire's
emitted light directly to T (the blackbody is already T-based) rather than to I,
for natural flickering. A render-side note that pairs with Q4.

### Locked still-air values (fire-tuning branch `aedca7f` — goldens NOT rebased)
`[physics.fire]` k_grow=0.04 · k_die=0.02 · fuel_ref=40 · k_wind_strip=0 ·
wall_damage=0.028   `[physics.combustion]` fuel_per_o2=0.045
→ crate peak I≈0.61 @ 2.85 min, burnout 9.12 min (still air).
- **★ TODO (Erik, tonight): rebase the fire / digest / golden tests** for these values.
- SOLID (ramp): `k_grow` / `k_die` / `k_wind_strip` / `fuel_ref`.
- COUPLED to the Fable O₂ pass (may move if `burn_rate` changes): `fuel_per_o2` /
  `wall_damage` / the burnout number.

## 8. THREAD TRACKER — Erik's simple todo (created 2026-07-24, tick as you go)

Order matters only where numbered; 3a/3b run in parallel.

1. ☑ **Design docs BLESSED (Erik, 2026-07-24)** — all three
   (`continuous_o2_law` · `sky_exchange` · `ema_sponge`). NOTE: the P1b
   zombie-smolder fix is doc-gated on an EXPLICIT yes — if the blessing
   covers it, say so in the o2-law build chat.
2. ☑ **Fire-tuning branch PARKED** (Erik, 2026-07-24) — no interim golden
   rebase; one rebase at the end of the joint re-tune.
3. Build status (2026-07-24):
   - 3a. ☑ **sky-exchange BUILT+PUSHED** (origin/sky-exchange 15afdb9, all
     gates green incl. lockstep on Ada; host-side pass + dirty-flag mask).
     **Erik's call: the standalone HUMAN-TEST is REPLACED by NUMBERS** at the
     step-4a harness run; **τ=60 ACCEPTED as default** until a reason appears.
   - 3b. ☑ **o2-continuous-law BUILT** (3 commits, gate-a 14/14, CUDA
     lockstep tol-0, P1b in; branch-red tests enumerated in its handoff doc —
     by design, rebaseline at re-tune). **LOCAL-ONLY → Erik: tell that chat
     to PUSH the branch** (safekeeping; push ≠ merge).
   - 3c. ☐ **smoke-single-source** ← `smoke_single_source_design_2026-07-24.md`
     (Q6; blessed). Spawn once the 4a integration line exists — stack on it.
   - 3d. ☐ **thermal-mass axis** ← `thermal_mass_axis_design_2026-07-25.md`
     (§9.5 regression fix; blessed 2026-07-25). **BLOCKS the 4b re-tune** —
     spawn NEXT, on the integration line. Erik's manual loop resumes after
     P3 updates the TUNE defaults (k_fire_heat ≈ 225 scale).
   MERGE NOTE (all three): goldens move by design and ride ONE deliberate
   rebase at step 5 — branches converge on the integration line, NOT
   individually onto main.
4. Integration + numbers:
   - 4a. ☑ **NUMBERS RUN DONE 2026-07-24** (fire-o2-integration @ 6e74c29,
     sponge-safe 84×40 bench, deep crate (12,21), sky τ=60). Findings:
     (i) deep crate SUSTAINS (peak I 0.71 @ 7.2 min) — density trap closed ✓;
     (ii) the O₂ *plane* collapses but by THERMAL DECOMPRESSION (room T
     ~2.3× ambient from ONE crate → N_total 0.44 at pinned P) — mole
     fraction X holds 0.19–0.21; **RULING (Fable): O₂-health metric is X,
     not the density plane**; (iii) sky-exchange active + conservation-safe
     but near-inert on an open bench (nothing to correct) — **its real
     validation moves to the Phase-3 ENCLOSED choke bench**; (iv) baseline
     for re-tune: ramp-to-90% 3.6 min, peak 0.71 @ 7.2 min, flame T ~8000
     (cooling-limited — rails without starvation), NO burnout (~85 min
     extrap., I=0.10 smolder tail). Harness sky-plumbing is LOAD-BEARING
     (sky_tau_s default-dormant) — **Erik: tell that chat "commit the
     harness changes to fire-o2-integration"**.
   - 4a'. ☑ **MERGES DONE (2026-07-24):** `sky-exchange` MERGED to main
     (4db60ff, re-verified 1680 green). `o2-continuous-law` PUSHED, stays
     PARKED (branch-red by design); tuning runs in the integration worktree
     via `--set`, merge only after blessed values + the ONE golden rebase.
   - 4b. ☐ **the ONE re-tune** (burn_rate FIXED 0.02) — **Erik's targets
     (2026-07-24): peak I ≈ 0.5 (0.4–0.6 acceptable), peak @ ~3 min,
     burnout 6–8 min.** Order:
     (1) ramp `k_grow/k_die` (ratio→peak 0.5; magnitude→peak @ ~3 min),
     (2) burnout `wall_damage` — **BLOCKED 2026-07-24 then RULED (Fable):
     the zombie smolder is a DESIGN BUG — ignition is level-triggered
     (per-tick `max(I,0.1)` floor) where it must be EDGE-triggered. Fix
     (implemented in the integration chat): `ignition_armed` bool plane,
     seed-once + disarm, re-arm only when T cools below ignition_temp; the
     floor removed (ignition never touches I>0). "Burnout" REDEFINED =
     fire-death time (leftover wall_hp = charred remains, correct). Then
     re-tune `wall_damage` for 6–8 min lifetime (expect 0.028–0.15), (3) **heat balance `k_fire_heat` vs `COOL_SHIFT`** — REALISM
     targets (Fable, from the game scale K=293+2·T): flame-tile steady
     T ≈ **400–500 game** (≈900–1300 K, a real wood/crate flame; also fixes
     the blackbody look — orange, not blue-white), far-field (>10 tiles)
     room-T rise ≤ ~20 game, room N_total ≥ ~0.9 all burn (no
     decompression). If sustain FAILS at realistic T, the culprit is the
     `fire_T_ext`/`fire_T_span` hot-gate window — report, don't crank
     k_fire_heat back up. (4) `X_ext=0.13` + sky `τ=60` ACCEPTED (Erik).
     Iterate (1)↔(3) once — they couple through the hot gate. You bless
     each value. THEN: spread (pair) → enclosed O₂-choke (sky-exchange's
     real test) + burst → smoke look (needs 3c).
5. ☐ **ONE golden rebase** (written rationale) → **HUMAN-TEST** (you play) →
   merge fire-tuning to main. Locked-values TODO from §7 dissolves into this.
6. ☐ (anytime, before the wind phase) Spawn Opus build: **EMA sponge** ← its
   design doc.
7. ☐ Back to Fable, next §7 questions in order: **Q6 soot(O₂)** → Q5
   chemistry lump → Q4 I-vs-T → Q7 fire light.

Parked, unrelated to this arc (don't lose): B2 smoke-honesty HUMAN-TEST
(fire-b2-smoke-honesty branch) · Arc C editor C10 drive-test · main push call
(control-modularity stacking).

## 9. THE FIRE CHAIN — one page, chronological (written 2026-07-25 for Erik's
manual tuning loop; supersedes §2 as the tuning reference)

### 9.1 Life of a fire (post o2-law + edge-trigger fix)

1. **HEATING.** External heat raises a flammable tile's `temperature` T
   (game units; K = 293 + 2·T, ambient 0 = 20 °C). Heat arrives by
   conduction and by burning neighbours' heat rays; it leaves by conduction
   and ambient cooling (`COOL_SHIFT`).
2. **IGNITION (an EVENT, once).** When an armed tile crosses
   `ignition_temp` (wood 300 = 620 °C) with O₂ fraction > `o2_frac_ext` and
   fuel > 0 → intensity seeded `I = 0.1`, tile disarmed. Re-arms only after
   cooling below `ignition_temp`.
3. **GROWTH (the logistic).** `dI/dt = k_grow·avail·hot·I(1−I) −
   k_die·(1−avail·hot)·I` where `avail = F·o2f`, `F = wall_hp/fuel_ref`
   (fuel fraction), `o2f = clamp01((X−0.13)/(0.21−0.13))` (X = local O₂
   MOLE FRACTION), `hot = clamp01((T−fire_T_ext)/fire_T_span)`.
   Equilibrium `I_eq = 1 − d/a`: the k_grow:k_die RATIO sets peak height,
   the magnitude sets ramp speed.
4. **WHILE BURNING, each tick:**
   a. **Heat out:** 8 rays deposit `k_fire_heat·I`, reach `range_base +
      range_per_intensity·I` → heats own tile (sustains `hot`) + neighbours
      (the SPREAD path).
   b. **O₂ in:** consumes `burn_rate·I·o2f` per open neighbour site
      (`burn_rate = 0.02`, Huggett-anchored — THE consumption dial) →
      heat `H_fuel`/O₂, soot, inert. Local X falls = true vitiation.
   c. **Fuel out:** `wall_damage·I` hp/s (the flame) + `fuel_per_o2` per O₂
      burned (the smolder coupling — NOT an O₂-consumption dial).
   d. **Sky refill** (planetside, sky-connected): X relaxes to 0.21, τ=60 s.
5. **DECLINE & DEATH.** Fuel falls → F falls → I_eq falls → I declines;
   below `I_min` it snaps to 0 and STAYS dead while hot (edge-trigger).
   Leftover `wall_hp` = charred remains — **"burnout" = fire-death time.**
6. **RE-IGNITION** only if the tile cools below `ignition_temp` and is
   re-heated later.

### 9.2 ★ Structural pre-fix found 2026-07-25 (Erik's k_fire_heat sweep):
**the cold-start gap = the §2.3 ordering wart, now load-bearing.** A tile
ignites at T=300 but `hot` only opens at `fire_T_ext=350` — the seed must
survive on `hot≈0` (pure death term) while its own tiny 0.1·k_fire_heat
deposit tries to climb 50 units. At low/realistic k_fire_heat it can't →
instant snap-out ("self-collapse, T-gated"), which is what the sweep shows.
**Fix (physical, do FIRST): `fire_T_ext` must be BELOW `ignition_temp`** —
established flames sustain below the ignition threshold in reality. Set
`fire_T_ext ≈ 250` (793 K — ember-sustain region), `fire_T_span ≈ 100`.
Then a freshly ignited tile (T ≥ 300) starts with `hot ≈ 0.5` and can climb.

### 9.3 Dial map — what to tune, in this order (Erik's loop)

**ANCHORED — set, verify, do not tune:**
- `burn_rate = 0.02` (Huggett 1980, 13.1 MJ/kg O₂ + the ceiling_h column) —
  O₂ draw per site. **This is the "O₂ disappears too fast" fix; verify each
  run: far-field X ≈ 0.21 throughout an open-field burn.**
- `o2_frac_ext = 0.13` (Peatross–Beyler 1997 extinction limit).
- `fuel_per_o2`: **RESTORE to 0.7 (wood stoichiometry).** The 0.045 hack
  existed only to compensate burn_rate=1.0; with 0.02 anchored, the
  physical value costs ~0.008 hp/s — negligible and honest.
- `sky_tau_s = 60`, `o2_frac_amb = 0.21`, `ignition_temp` table (§3).

**TUNE, in this order (one at a time, iterate the pair 2↔3 once):**
1. **Structure:** `fire_T_ext = 250`, `fire_T_span ≈ 100` (§9.2). One-time.
2. **Thermal loop:** `k_fire_heat` (down from 1600; expect O(10–60)) vs
   `COOL_SHIFT`. TARGETS: flame-tile steady T ≈ **400–500 game**
   (900–1300 K — a real wood flame; orange blackbody), far-field (>10
   tiles) rise ≤ **20 game**, room N_total ≥ **0.9** (no decompression).
   Tune this BEFORE the ramp — T is the substrate every gate reads.
3. **Ramp:** `k_grow`/`k_die`. TARGETS: peak I ≈ **0.5** (0.4–0.6 ok),
   peak at ~**3 min**. Ratio → height, magnitude → speed.
4. **Lifetime:** `wall_damage`. TARGET: fire death **6–8 min**, charred
   remains expected (hp>0 at death is correct).
5. **Verify anchors:** near-flame X dips (vitiation visible), far X ≈0.21,
   sky τ behaves. No tuning — just confirmation.
Later phases (unchanged): spread pair (`range_*`, k_fire_heat interplay) →
enclosed choke bench (sky-exchange's real test) + burst → smoke (needs 3c).

### 9.4 Why manual: the chain is one coupled dynamical system — T gates I,
I makes T, both eat fuel and O₂. Sweeping one dial blind moves every
equilibrium; the loop that converges is: set structure → fix the THERMAL
operating point → then shape the I-dynamics on top of it → then set the
clock (fuel). Erik drives; the script's constants block mirrors this order.

### 9.5 ✅ REGRESSION — **FIXED** (built + gated 2026-07-30; awaiting Erik's
HUMAN-TEST). *Rewritten at P3 close; the "open question" version is superseded.*

**What it was.** The crate took the GAS thermal branch. `[materials.furniture]
permeability = 0.5` → NOT `solid` → `COOL_SHIFT` never applied to a burning
crate, furniture κ=0 meant no conduction either way, and the crate's
"temperature" was hot gas the fire's own EOS plume advected away (280 → a
~90–110 shelf in seconds, measured at −21/−35/−33 game per tick). No cooling
dial controlled it. Diagnosed 2026-07-25; **blessed by Erik the same day as a
REGRESSION, not a question** — canon 06:77-91 had designed ignition temperature
to live on objects ("air temperature would ignite nothing and advect
everything — the wrong behavior"), and the EOS-era unified field keyed the
medium masks on `solid` (= the FLOW axis), so furniture fell into the gas
regime by accident. The `is_wall` error class, one axis over.

**The fix, as built.** The missing axis: per-material **`thermal_mass`** (`> 0`
→ solid thermal regime, and the value IS the convert divisor, power-of-two);
furniture joins the walls; `permeability` (shield-not-seal) UNTOUCHED. A
derived `thermal_solid = (thermal_mass > 0)` mask, built on the one
structural-rebuild seam, now routes the thermal medium.

| commit | patch |
|---|---|
| `f5e9aa3` | P1 — CPU: `thermal_solid` replaces `solid` at the six medium-test sites in `temperature_solver.cpp` |
| `312e984` | P2 — CUDA: the six twins + the resident path, lockstep tol 0 |
| `6f57762` | P-EOS — the EOS pass: both T-writes skip thermal_solid, T-only occluder, `cmask` untouched; combustion deposit re-routed to the object divisor |

Docs: design `docs/thermal_mass_axis_design_2026-07-25.md` → build addendum
`..._build_addendum_2026-07-30.md` → escalation `thermal_mass_eos_escalation_2026-07-30.md`
→ **ruling** `thermal_mass_eos_ruling_2026-07-30.md` → **bench report**
`thermal_mass_axis_bench_report_2026-07-30.md`.

**The rule that came out of it** (ruling §1, now the thing that stops this
recurring): *on `thermal_solid` tiles `temperature[]` is OWNED by the
TemperatureSolver — deposit-convert, conduct, COOL_SHIFT. Every other system is
a READER.* Measured in the live path: the EOS pass moves the crate's T by
**0.000** game/tick, every run.

**What this un-does in this document.** §9.5's old "verified starting set"
(`fire_T_ext = 60`, span 60, `k_fire_heat = 12`, COOL_SHIFT inert) was a
description of the *broken* regime and is now **dead** — do not tune from it.
§9.2's below-ignition rule **stands** and is back on its original physical
footing: `fire_T_ext = 250`, span 100, below furniture's `ignition_temp = 280`.
§9.3's dial order is **unchanged** — structure → thermal → ramp → lifetime →
verify anchors — and its step 2 finally means what it says, because COOL_SHIFT
is now a real dial for the crate and κ=0 makes it the *only* loss channel
(ruling A5, deliberate: one clean channel per dial). One stale parenthetical
there: step 2's "expect O(10–60)" for `k_fire_heat` was inherited from the
broken-regime `k_fire_heat = 12` and is meaningless on its own now — the value
only has meaning **paired with a COOL_SHIFT**, via the equilibrium below. Read
the starting values from "Start here" at the bottom of this section instead.

**The equilibrium is now exact** (bench report §3, ±1 % measured at three
operating points):

> **T\*(I) = k_fire_heat · I · 2^(COOL_SHIFT − heat_inv_shift)**,
> `heat_inv_shift = log2(thermal_mass)` = 3 for furniture — so at
> `COOL_SHIFT = 5`, `T* = 4 · k_fire_heat · I`.

**Two corrections to what the arc had on record**, both from the P3 bench:
- The 0.871 analytic ratio P-EOS reported is a **12-second transient**, not a
  steady-state deficit. At equilibrium it is 0.995–1.009. Do not carry it as a
  fudge factor.
- The design's expected new operating point **`k_fire_heat ≈ 225`** is
  arithmetically right and **dynamically dead**: measured, it snaps out at tick
  1. §9.3 step 2 does not start there.

**★ What is now OPEN — the one thing the fix exposed rather than solved.**
Deposit is linear in I and loss is linear in T, so `T*` is linear in I, and the
hot gate opens at `fire_T_ext`. Hence a critical intensity

> **I_crit = I_peak · fire_T_ext / T_flame**

below which the gate closes and the fire self-collapses. At §9.3's own targets
(flame 450 @ peak I 0.5, `fire_T_ext` 250) that is **I_crit = 0.278** — nearly
3× `ignition_seed = 0.1`. So at the target thermal point the fire can neither
ignite from the seed nor burn down gracefully; it is fenced into `I > 0.278` at
both ends. This is §9.2's cold-start gap generalised, now in the object regime.

Four levers, all measured (bench report §4.3). Three relocate the cliff —
COOL_SHIFT ≈ 12 (works, but COOL_SHIFT is **global**: every wall's cooling
e-fold goes 1.3 s → 171 s); `ignition_seed` into the band (ignites at flame
449, then dies at 48 s); `fire_T_ext` below `T*(I_seed)` (flame 450, lives
208 s, but 90 game = 200 °C is not a defensible extinction temperature). Only
the fourth removes it — making the deposit non-linear in I — and that is a
**model change, not a dial**. **Erik's call, at the joint re-tune.**

**Start here (§9.3 hand-back).** `tools/fire_tune_loop.py` now ships the best
measured set as its TUNE defaults, with COOL_SHIFT exposed as a dial and the
ALT branches inline:

```
conda run -n data python tools/fire_tune_loop.py     # edit TUNE, run, read, repeat
```

Defaults: `fire_T_ext 250 / span 100` (step 1, done) · `k_fire_heat 2.2` +
`COOL_SHIFT 12` (step 2) · `k_grow 0.35 / k_die 0.06` (step 3) ·
`wall_damage 0.083` (step 4). Measured at 900 s: flame plateau **414** (1120 K)
✓, peak @ **144 s** ✓, room N **1.000** ✓, far X **0.199** ✓, charred remains
✓; misses are peak I 0.331 (low), death 5.5 min (target 6–8), far rise 21.4
(target ≤20) — all three in **your** steps 3–4. Tune step 3 first (raise
`k_grow`/`k_die` together toward the 0.6/0.1 sibling, which gives peak I 0.411
but at 61 s), then step 4 (`wall_damage` down to stretch death to 6–8 min).
Then decide the COOL_SHIFT question above — that one is a design call, not a
dial turn.

*Forward notes, unchanged:* units ignitable later (armor-dependent ignition
temps, unit-environment system, own mechanics design); movable-furniture
migration path. *Not merged:* feel-adjacent → HUMAN-TEST gate; no golden
rebased — it rides the joint re-tune's ONE deliberate rebase.

## Session log
- **2026-07-22** — Doc created. Read the current model + values from code.
  Corrected the temperature-scale understanding (game units + `293 + 2·T`
  blackbody map, `k_temp_to_kelvin=2`, *not* Kelvin/Celsius). §4 order approved.
  **Decision #1: KEEP game units** (Erik — scale is clever, ignition temps
  good; a brief "absolute Kelvin" call was reverted after the scale was
  clarified). §5.2 added: first-principles single-crate ramp (~3.6 s today) +
  burnout (~33 s today) — both ~50–100× too fast vs Erik's 3–5 min / 5–10 min
  targets; crate is fuel-limited to I_eq=0.5. Next: burn-bench + real-engine sim
  to confirm, then tune `k_grow`/`k_die`.
- **2026-07-23** — Harness built (`tools/fire_timing_harness.py`, §5.3);
  still-air burnout ≈39 s confirms too-fast. ★ Ramp is COUPLED — slowing k_grow
  alone STALLS (plume self-blow-out via `k_wind_strip` on the fire's own wind) +
  fuel 40 s cap. Fix (Erik's convective-cooling idea, pulled forward):
  `k_wind_strip=0` + slow fuel → THEN k_grow. Phase-1 order revised (§4). Two
  model quirks logged (O₂-thins-at-flame → crate must sit by the ambient ring;
  zombie-smolder → re-ignites past fuel-out). Coupled experiment running for the
  target dial values (peak@2–3 min, burnout@5–10 min). Aside: B2 merged to LOCAL
  main + green (icon flake fixed); not pushed (concurrent control-modularity arc
  stacked on main — Erik's push call pending).
- **2026-07-23 (Fable O₂ session)** — **§7 Q2 RESOLVED** (full block in §7): BC
  verified correct + identical harness/editor; depletion = consumption budget +
  no-suction-while-burning + default sponge over most of the bench; asymmetry
  explained (crate-at-ring + sponge-free mid channel), not a wind bug.
  Decisions (Erik): **Option A sky exchange, composition-swap** (O₂/N₂ first,
  smoke-λ deferred to B2 look, T covered by COOL_SHIFT); **EMA high-pass
  sponge** chosen for windy levels — own design-gated item, not blocking
  tuning; bench-sizing rule adopted. `ceiling_h = 2.5` m found canonical
  (`[physics.water]`, "one constant for the air column") → Q3 realism preview:
  burn_rate ~50–90× too fast, expect O(0.01–0.02). Next: Q1.
  LATER SAME DAY: **EMA sponge design doc written**
  (`docs/ema_sponge_design_2026-07-23.md` — gates, patch plan, escalation
  triggers; own branch, Opus build). **burn_rate=0.02 harness-verified** — it
  UNRAVELS the locked flame (see ★ Q3 MEASURED in §7): peak 0.81, T rails at
  16000, burnout ~23 min; the cut is RIGHT but lands inside Q1 as one re-tune;
  config untouched.
- **2026-07-24 (Fable)** — **Q1 DESIGNED + Q2 build spec'd**, both as Opus
  kickoff docs: `continuous_o2_law_design_2026-07-24.md` (verdict: continuous
  IS more realistic — Peatross–Beyler linear law; gate moves from absolute
  density to MOLE FRACTION with extinction dial `o2_frac_ext`≈0.13, killing
  the v2.4 density-trap rescale saga + quirk #1; draw becomes
  `burn_rate·I·o2f`; burn_rate=0.02 lands inside; optional P1b zombie-smolder
  fix awaits Erik's nod) and `sky_exchange_design_2026-07-24.md` (sky_mask
  flood fill, per-tick composition swap at fixed N_total, `[ambient]
  sky_tau_s`≈60, sky_flux conservation rail). §8 THREAD TRACKER added —
  Erik's follow-along todo. Joint re-tune AFTER both builds merge.
- **2026-07-24 (Fable, later)** — Erik BLESSED all docs, PARKED fire-tuning
  branch, SPAWNED both builds. Then **Q6 settled** (see ★ in §7): Erik's
  architectural call — delete smoke source A, combustion soot becomes sole
  fire-smoke source with the O₂-dependent yield + fuel-mass share; design doc
  `smoke_single_source_design_2026-07-24.md`; builds stacked after
  `o2-continuous-law`; §8 gains 3c + the integration-line merge note.
  Clarified en route: Erik's dirty-Planck flame-darkening idea is a separate
  render experiment, not the Q6 sim law. Remaining §7 queue: Q5 → Q4 → Q7.
- **2026-07-25 (Fable)** — Erik's manual tuning loop delivered
  (`tools/fire_tune_loop.py` + auto Kelvin plot + VSCode F5, integration
  worktree) with warm-seed harness fix. First runs exposed the §9.5
  structure; Erik asked the right question ("shouldn't we fix the furniture
  first?") → canon dig → **§9.5 reclassified REGRESSION** (medium masks on
  the flow axis) → **thermal-mass-axis design blessed**
  (`thermal_mass_axis_design_2026-07-25.md`, tracker 3d, blocks 4b).
  Tuning paused until it lands. Units-ignitable (armor-dependent) noted as
  future mechanics design.
- **2026-07-24 (Fable, evening)** — 4a numbers run landed (see §8 4a for the
  findings + rulings): density trap closed, decompression ≠ vitiation (metric
  = X), sky-exchange near-inert on open bench (validate at Phase-3 choke),
  heat balance (`k_fire_heat`/`COOL_SHIFT`) PROMOTED into the single-crate
  re-tune — one crate cooked an 84×40 room to 2.3× ambient. Merge calls:
  sky-exchange → main now (behavior-free); o2-law stays parked. 4b re-tune
  order restated. 3c smoke build unblocked (integration line exists).

**Appended 2026-08-14 (supersession note).** Any ×2 game-T→Kelvin map referenced
above is superseded by the unified canonical map in
`[physics.temperature_scale]` (`K = 293 + 3·T_game`; EOS pressure calibration
keeps a named, deliberate exception at `eos_t_amb_k = 290`). See
`docs/temperature_scale_unification_design_2026-08-13.md`.
