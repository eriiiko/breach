# Breach — Fire/Smoke Physics + Beautiful Rendering

## BRAINSTORM / PROPOSAL — NOT CANON

> This document fuses two proposals (A: black-body soot/combustion; B: beautiful-rendering roadmap) into one
> coherent plan, cross-checked against the shipped engine docs (`05_smoke.md`, `06_temperature_and_fire.md`,
> `08_ray_engine.md`). It is a **brainstorm for Erik to decide on**, not an engine chapter. When a slice is
> picked, it gets written up properly as a `docs/architecture/engine/05` + `/08` chapter update with a
> `Depends on:` header and impl-status, per the design-docs-are-canon rule.

---

## 0. ADDENDUM — The ideal-gas reframe (the high-ambition direction)

> Added 2026-06-28, second brainstorm session with Erik. This addendum captures a direction that
> **partly supersedes §2–§3 below**: rather than bolt black-body emission onto the current pressure model,
> repurpose the pressure field via the ideal gas law so that temperature, pressure, shockwaves, fire, and
> the fireball lifecycle all fall out of one coherent thermodynamic model. Erik: "very happy with where we
> are, but one level up in quality starts to look like something that could make money." Still a
> BRAINSTORM — collection phase, critique deferred. Prototype in Python (float) first; port to C++/CUDA +
> fixed-point only if it earns its keep.

### 0.1 The reframe: pressure becomes density; pressure is derived

Ideal gas law `PV = nRT`. Per tile the volume `V` is fixed, so:

- **Stop storing pressure as the primary state. Store `N` = particle/mass density per tile** (Erik's
  "nR per unit area") — the quantity actually *conserved* and advected.
- Add **one** new field: a gas **temperature `T`** that now lives *in the air*.
- **Derive pressure every step:** `P = (R/V)·N·T ≡ C·N·T` — pressure = const × density × temperature.
- Wind/force is `−∇P` as today; continuity `∂N/∂t + ∇·(N·u) = 0` conserves gas mass.

The keystone. Costs **one extra field** (temperature) and turns the pressure field into a density field.

### 0.2 Dalton: all gases share ONE temperature and ONE velocity — only one new field total

Resolves the "per-gas or shared?" judgement call. All gases in a tile are one well-mixed air parcel:

- they **share** the temperature `T` and the bulk velocity/wind `u` (one each, total);
- each gas keeps **its own density slice `N_i`** (already present as `gmap.gas`);
- total pressure is the sum of partial pressures (**Dalton**): `P = C·T·Σ N_i`;
- per-gas **molar mass `M_i`** is a *constant* from the table (drives buoyancy/settling), not a field.

So the EOS is paid **once**, not per gas — exactly Erik's hope ("one extra field, temp; let the gases
ride it"). Heavy/light differential motion (poison sinks, steam rises) is a small per-species drift on
top of the shared advection.

### 0.3 What the reframe unlocks "for free"

- **Shockwaves from energy, not injected pressure.** A detonation dumps *heat* into a few tiles → `T`
  spikes → `P = C·N·T` spikes → `−∇P` blows gas outward = the blast. You add energy, not phantom mass, so
  **the atmosphere never artificially builds up** (the current pressure-injection artifact). This was
  Erik's original motivation.
- **Adiabatic expansion cooling = the fireball cooling arc, for real.** A hot parcel that expands does
  work on neighbours (the `−P(∇·u)` term) and cools → reddens → fades to soot. The
  `fireball → plume → black smoke` colour arc becomes a thermodynamic *consequence*, not the hand-tuned T⁴
  decay of §2/§3.
- **Baroclinic vorticity = real curl / mushroom clouds.** The vorticity source term `∇(1/N) × ∇P` fires
  wherever density and pressure gradients misalign — exactly at fire boundaries and blast fronts (the
  physical origin of rolling smoke and mushroom caps). **Curl-noise demotes** from "swirl generator" to
  optional sub-tile garnish; **MacCormack/BFECC earns its place doubly** — its job becomes *preserving*
  the real vorticity the EOS generates instead of diffusing it to mush.
- **A better O₂ proxy.** `N` (actual amount of gas) beats pressure as the oxygen proxy: a hot region can
  have high `P` but little gas (low `N`) — precisely when a fire should starve. Decompression,
  fire-suffocation and the breach outrush collapse into one story (breach → `N` drains to vacuum → `P`
  falls → no `N` ⇒ no O₂ ⇒ fire dies, and `−∇P` *is* the outrush).
- **The §2 "crux" dissolves.** The render-only `glow_temperature` hack is no longer needed: the gas
  temperature is **real and load-bearing** (it drives pressure), and black-body emission is simply "hot
  gas glows" off that same `T`. One field, two readers (the EOS and the §6.2 LUT).
- **Possible unification of the two pressure fields.** `atmosphere + wave_p` may collapse into one
  coherent `(N, u, T)` compressible system — wind, wave, plume and shock all the same equations rather
  than a slow-equalization field with an acoustic field bolted on. (Sound speed `c = √(γRT/M)` then
  depends on `T`, so the blast front steepens into a shock naturally — a CFL item for the prototype.)
- The first doc's optional **"expansion→pressure tap"** (§4 item / decision 3) becomes **intrinsic** —
  the pop is just the EOS, no special tap.

### 0.4 Gases as distinct physical substances (taxonomy reframe)

Erik's rename intent, corrected: `white_smoke`/`black_smoke` were just *colour variants*. The new model
makes gases **physically distinct classes**, each a real substance with its own property *set*:

- **soot** (was black_smoke) — the black body. Strong broadband absorber ⇒ strong emitter when hot
  (Kirchhoff); emissivity ~0.8. Heavy, clingy, lingers. Trace `N`, huge optical/emissive weight.
- **steam** (was white_smoke) — water vapour. **No** black-body. Bright Mie scatter, **latent heat**
  (vaporizing cools / condensing releases heat), condenses to water. Light, spreads fast. Water↔steam
  phase change owned by the water effort (Fable) — we only *emit* steam as a source.
- **poison** — heavy (high `M`), pools/settles low, persistent, damage-over-time.
- **combustion / fuel gas** — the flammable substrate; ignites → soot + steam + energy.

Molar mass `M_i` becomes a real, distinguishing per-gas parameter (buoyancy, pooling, blast response) — a
new axis of behaviour that drops out of the EOS. The `fuel → fireball → soot` lifecycle, retold
thermodynamically: *fuel + O₂ ignites → converts mass to soot + steam, dumps energy → `T` spikes → `P`
spikes (the pop) → gas expands and adiabatically cools → reddens → cold absorbing soot drifts and (heavy)
settles.* Every arrow is physics, not script.

### 0.5 2.5D multi-layer smoke — what it is and why it needs the EOS

- **Flat 2D (today):** one grid seen from above; "up" has no gradient, so smoke can only ooze sideways —
  it can never rise, billow, or form a column.
- **2.5D:** keep the 2D top-down grid but stack a *small* number `K` of horizontal layers in z
  (`gmap.gas[gas, z, y, x]`, K≈2 → ~8–10). Each z-slice is a full 2D sim sharing the solver/wind. Not a
  deep voxel volume — most of the *look* of 3D smoke for a small ×K of the 2D cost.
- **Buoyancy = density-driven vertical exchange between layers.** "Rise" becomes "transfer gas to the
  layer above when this layer is less dense than it." The **EOS provides the density** that drives it
  (hot ⇒ low `N` at fixed `P` ⇒ buoyant); molar mass gives the differential (poison sinks, steam climbs).
- **Why it looks 3D:** the camera composites *down the column* (vertical optical depth → height becomes
  visible: a tall dense plume vs a thin wisp differ by *structure*, not one number); upper layers shadow
  lower ones, and the hot base glows up through cooler soot above (reddened for free).
- **Shared heightmap clips layers** = emergent tactics: a *low* wall blocks the floor layer but smoke
  pours *over* it in the upper layers, no new code.
- **The home for "rise and cool":** buoyancy (rises) + adiabatic expansion (cools) + LUT (reddens) = a
  rising, curling, cooling column with a bright base and a grey cap. Tier-D big lift (cost ×K), gated
  behind sparse-tile sim and the CUDA water port (S3).

### 0.6 Raycaster + z-shadows without changing the raycaster (the key factoring)

**Light transport separates by direction**, and the two directions differ wildly in difficulty:

- **Horizontal (x–y) is 2D-hard** — light must route *around* walls/doors → needs the DDA march. The
  **raycaster stays a single-plane 2D primitive, unchanged** (deposit-only, one-thread-per-ray,
  bit-identity heat all preserved).
- **Vertical (z) through a few stratified layers is 1D-easy** — straight up/down a column is a
  Beer–Lambert running product `exp(−Σ τ above)`. No routing, no DDA. A **sweep**, not a march.

So z-direction shadows are **not** placed in the raycaster (that would force a 3D DDA and break the
contract). The pipeline:

1. **Raycaster — single plane, unchanged.** Cast horizontal light/heat/glow once on the floor (or a
   column-collapsed gas field). Bit-identical to today.
2. **Vertical sweep — new, trivial.** One downward pass per column attenuates overhead/ambient light layer
   by layer = the z-shadow (top of plume bright, base dark; upper smoke shadows lower). This is exactly
   §6.1's "3-tap self-shadow", applied down the explicit z-stack.
3. **Composite — down-the-column integration** (camera looks down): front-to-back emission + transmission
   over the K layers = both the shading and the final pixel.

Corollaries:

- **Heat stays single-plane (floor)** — ignition/unit-damage live on the floor — so the fixed-point,
  bit-identical heat cast never goes multi-layer. **2.5D adds zero determinism risk**; only render-side
  light/smoke gains layers.
- **"Several z-dims" is a loop, never a kernel change.** If flat horizontal lighting looks wrong, invoke
  the *unchanged* kernel per layer (×K). The raycaster is one-z by nature; you choose how often to run it.
- **Accepted cost:** misses true *diagonal* cross-layer light bleed (a low light into a horizontally
  offset high plume). Second-order, rarely visible top-down, same spirit as the existing
  dominant-direction `light_dir` approximation. Optional garnish: let the vertical sweep spread slightly.

### 0.7 Process / philosophy for this direction

- **Prototype in Python first (all float, fast iteration), judge by eye.** Does a baroclinic-curling,
  energy-driven, adiabatically-cooling fireball look gorgeous and play well? If yes, port to C++/CUDA and
  make it deterministic; if not, it is a cheap spike lost.
- **The infrastructure exists.** C++ solvers for atmosphere/wave/smoke/fire/temperature/water/raycaster
  (`cpp/src/`); CUDA kernels for all of them already exist (the S3+ port is well underway); and
  **`cpp/src/fixed_point.h` already exists** — which answers Erik's standing "make a class out of the
  Q16.16 math?" comment: it already is one.
- **Don't over-lock into the current design.** Willing to change the pressure model; may not do it for a
  while. Determinism is a *deferred, solved pattern* (`fixed_point.h` + the CUDA bit-identity discipline),
  not a wall.
- **MacCormack/BFECC anti-diffusion** confirmed a keeper (pairs with curl-noise *and* baroclinic-vorticity
  preservation).
- **Ambition bar:** "one level up in quality = starts to look like something that could make money."

### 0.8 Open threads for next session (not yet discussed)

- How combustion actually drives the new `(N, T)` fields in the prototype (burn rate, yields, energy
  release numbers).
- Choosing the compressible/EOS scheme for the prototype (Euler + artificial viscosity, thermal
  Lattice-Boltzmann, Kwatra-style stable compressible flow, Feldman/O'Brien suspended-particle explosions,
  divergence-control for fire) — a focused research workflow was offered, not yet run.
- Whether `atmosphere + wave_p` actually collapse into one `(N, u, T)` system, or stay split.

### 0.9 Determinism strategy for the EOS fluid (from the determinism research)

A companion research workflow (2026-06-28) asked whether the fixed-point apparatus can be avoided — e.g. by
**coarsening gameplay decisions** to a coarser float resolution. Verdict: **no.** Coarsening is a longer fuse,
not a defusal — desync is born upstream in the arithmetic and chaos re-amplifies it, so coarsening delays
divergence only *logarithmically* (`t* ≈ (1/λ)·ln(Q/δ)`; 1000× coarser bins buy ~`6.9/λ` extra ticks). Full
report: `docs/determinism_without_fixedpoint_research.md`. The load-bearing implication **for the EOS
direction** (the two threads pull opposite ways and this is the reconciliation):

- The EOS makes the fluid's `T`/`P` *central* — but a **chaotic field that feeds gameplay is the most
  expensive thing to make cross-machine deterministic.** So **do NOT make the chaotic `(N, u, T)` fluid
  authoritative.** Run it in **float, locally** per client (and same-box for ML training — where float is
  already deterministic once RNG + reduction order are pinned; no fixed-point needed there).
- Keep the **deterministic surface thin**: gameplay depends only on a few *scalar crossing-quantities* the
  fluid hands off — ignition flag, shockwave impulse, `water_depth` heat-sink, unit HP. Those few are
  integer/fixed-point; the gorgeous fluid stays render-only/local. (Precedent: Company of Heroes ran
  non-deterministic explosions over a synced sim.)
- **Design the fluid→gameplay interface as that thin scalar hand-off from the start** of the prototype, so
  the ambitious EOS fluid and tractable determinism coexist.
- The real lever is **SCOPE, not removal** of fixed-point. Keep the integer core (integer `atomicAdd` is
  strictly stronger *and* cheaper than any reproducible-float reduction). Add a per-N-tick **checksum**
  (detect-fast, not prove-forever). Heterogeneous cross-arch lockstep (CPU-client vs GPU-client, MSVC vs
  nvcc) is the only thing needing the full apparatus and is **deferrable** for a co-op/PvE game; the hardest
  artifact (CPU≡CUDA heat at tol 0) is already shipped. Keep host-baked transcendentals + `--fmad=false` +
  `/fp:strict` regardless.

---

## 1. Your direct questions, answered plainly

**Does temperature ride the wind?** No. By foundational design (`06_temperature_and_fire.md` §1) heat crosses
air **only as radiation** — rays write a per-tile `heat` deposit. `temperature` lives on **solids only**
(`conductivity = 0` on air) and is never advected. Temperature does not ride the wind, and there is no
air/gas-temperature field at all.

**Do gases ride the wind?** Yes. All gas density slices (`gmap.gas`: white_smoke/steam, black_smoke, poison,
teargas, fuel_gas) ride the **one shared wind field** (`wind = −grad(atmosphere + wave_p)`) via the same
batched diffusion + semi-Lagrangian transport (`05_smoke.md` §2, §6.2 — M1+M2 shipped).

**Is there an air-temperature field today?** No, and that is deliberate (the §1 "single foundational choice").
The only per-air-tile thermal quantity is the `heat` deposit — radiation **flux arriving this tick**, written
by the ray engine, read by the temperature solver + unit-damage, then **cleared end-of-tick**. It is not stored
warmth and it does not move with the air.

**Is black-body emission already specced?** Yes — and more completely than "half-specced". `05_smoke.md` §6.2
sub-note gives the **full** emission model: Planckian-locus chromaticity LUT (600 K dull red … 5000 K
near-white, clipped to sRGB) × a shifted Stefan–Boltzmann intensity ramp `I = ((T−T0)/(Tref−T0))^4`
(`T0 = 800 K`, `Tref = 2000 K`, clamp `I_max ~ 3–4`), × `smoke_density`, added **after** transmission and
**not** subject to the gas's own absorption (thin-slab). §6.1 step 4 restates it as a real-time fit. The gate
flag (`emits_when_hot`) and a `heat`-buffer temperature source are named. **The math is canon; only the wiring
is unbuilt.** What is genuinely missing is the *crux below*.

---

## 2. The crux: how does a drifting gas tile know it is hot?

> **Update (see §0):** if we take the ideal-gas reframe, this crux **dissolves** — the gas carries a real
> temperature `T` (load-bearing for pressure), so emission is just "hot gas glows" off that `T`, with no
> render-only `glow_temperature` field needed. The options below remain the right answer **only** if we
> keep the current pressure model. §0 is the more ambitious path.

Both proposals converge on the same problem and (mostly) the same answer. The §6.2 emission math needs a
per-tile temperature `T`. The doc says "drive `T` from the `heat` buffer". But the `heat` buffer is a
**cleared-each-tick transient with no memory** — a soot tile that drifted 5 tiles downwind has no record it was
ever hot, and `heat` does not advect. So reading `heat` directly gives you a fireball that is on **where the
fire/radiation is right now** and off everywhere else — no cooling, reddening, fading plume. That cooling tail
is half of the "smoke → fireball → black smoke" wish.

Three options (from Proposal A, sharpened):

- **(a) Drive glow from `fire` + the live `heat` deposit, no new field.**
  `T_emit = lerp(T0, Tflame, fire[tile])` (small blur so soot just off the flame still glows), optionally
  `+ k·dequant(heat)`. Zero new fields/transport/memory; render-only; respects every rule. **But no memory** —
  drifting soot goes instantly cold. Fireball-on/off, no arc. Cheapest.

- **(b) RECOMMENDED — a persistent, advected, render-only `glow_temperature` field.**
  One `float32` field on `GameMap` (explicitly **not** `gmap.temperature`, which is solids-only Q16.16). Per tick:
  *seed* it from the combustion burn term where fuel ignites; *advect* it on the **same shared wind** (it batches
  as one more slice into the existing semi-Lagrangian transport — free); *decay* it with the shifted T⁴ law so
  hot cores die fast and warm soot lingers. Feed it into the §6.2 LUT. This is the enthalpy/warmth proxy that
  physically *should* ride with the gas.

- **(c) Hybrid: (b) as the spine + (a)'s `max(glow_temperature, T0 + k·dequant(heat))` boost** so soot drifting
  *through* a fresh hot field (another fire, an explosion flash) brightens this tick before its own `glow_T`
  catches up. Realises the explosion-flash-lights-nearby-soot idea.

### Recommendation
Build **(b)** as the spine; keep **(c)**'s transient boost as an optional Tier-2 additive. Reject (a)-alone —
it cannot produce the cooling/reddening plume.

**Why (b) does NOT violate the no-air-temperature rule.** The rule's purpose (`06` §1) is to keep *gameplay*
heat transport radiation-only — ignition, wall-melt, unit-damage must come from rays into `temperature`/`heat`,
never from an advecting warm-air field. `glow_temperature` is a **perceptual byproduct field** that rides the
wind exactly like the gas densities and `smoke_glow` already do, and **never gates anything**. The iron rule
that keeps us honest: *`glow_temperature` is never read by ignition, damage, AI, or any sim threshold.* Naming
it `glow_temperature` (not `air_temperature`) and documenting "render-only, never a gameplay input" is the
guardrail. Float, render-only, downstream of gameplay → zero determinism cost, provided that discipline holds.

This also gives the **two-timescale split** that *is* the fireball→soot arc: fast T⁴ `glow_temperature` decay
(the flicker, the orange-core-to-dark-crown gradient) vs slow `black_smoke` density decay (the lingering pall).

---

## 3. The black-body soot + fuel_gas → fireball → soot lifecycle

A per-tile state machine **derived from thresholds** (not stored), layered over existing fields + the one new
`glow_temperature`. ~80 % is wiring of specced-not-built canon; ~20 % is the new render field + an optional tap.

```
fuel_gas  (drifts invisibly on the wind; NEW: apply the loaded-but-unused per-gas decay)
   │  [temperature ≥ ignition_temp AND pressure-as-O2 ok]   ← EXISTING fixed-point gate
   ▼
IGNITED — the NEW combustion term:
   burn          = min(fuel_gas, k_burn·dt)            # float rate
   fuel_gas     -= burn
   heat_deposit += burn·H_fuel                         # into the EXISTING Q16.16 heat buffer
   black_smoke  += burn·soot_yield
   steam        += burn·vapor_yield                    # combustion is a STEAM SOURCE (§5)
   glow_temperature = max(glow_T, T_flame)             # seed the NEW render field
   (OPTIONAL) wave_p/atmosphere += burn·expansion      # the fireball self-push "pop"
   ▼
FIREBALL — glow_temperature high → §6.2 LUT emits bright through the ray march;
           fire field is already a ray light+heat source (EXISTING, no special path) → room lit free
   │  [fuel depleted / glow_temperature cooling via T⁴ decay]
   ▼
PLUME — glow_temperature reddens then fades (the §6.2 shifted T⁴ ramp);
        black_smoke persists, advected on the wind; emission reads orange → red → dark
   │  [glow_temperature < T_glow0]
   ▼
BLACK SMOKE — emission → 0; pure absorbing soot; slow per-gas decay (NEW: apply it) /
              O2-keyed oxidation sink clears it; decompression vents it (EXISTING)
```

**Soot can hold temperature and shine — and it is the physically primary visible emitter.** A luminous yellow
flame *is* glowing soot, not glowing gas. Soot is a near-grey-body: cold soot looks black because it is a strong
broadband absorber, and by Kirchhoff's law that same strong absorber is a strong **emitter** when hot. So
Breach's single `black_smoke` slice legitimately has two behaviours from one field, gated by temperature:
absorbs-when-cold (the per-channel Beer–Lambert it already does) and glows-when-hot (the unbuilt emission term).
Emissivity is one scalar ~0.8 multiplier. The render line, inside the existing DDA march, front-to-back, per
emitting tile:

```
result_rgb += transmit_rgb · LUT(glow_temperature[tile]) · emissivity · black_smoke_density[tile]
              // weighted by smoke ALREADY in front (hot core dims behind cooler smoke)
              // BUT not attenuated by its own cell's absorption (thin-slab, §6.2)
// then, as today:
transmit_rgb *= trans_c
```

One LUT fetch + ~3 MADs per emitting tile, folded into the march that already computes per-channel `exp(−τ)`.
No new pass. The T⁴ ramp pushes hot cores HDR (>1) on purpose; the shipped ACES tonemap rolls them
red→orange→yellow→white **for free** — *if* ACES is per-channel (see §6, precondition).

**Steam source.** Combustion's `vapor_yield` product is steam (water in real fuels). Combustion only *emits*
steam; it must **not** assume steam ever condenses — that is the water effort's (§5).

### What is NEW vs what is merely consumed
- **NEW:** (1) the `glow_temperature` field + seed/advect/T⁴-decay; (2) the combustion term wiring
  `fuel_gas → heat + black_smoke + steam` (the §6.2 `flammable` column, loaded-not-built); (3) the black-body
  emission term in the march (the §6.2 `emits_when_hot` sub-note, specced-not-built); (4) applying the per-gas
  `decay` column (loaded-not-applied); (5) OPTIONAL the expansion→pressure tap.
- **EXISTING (consumed):** the ray march + per-channel optics; the `heat` buffer + temperature solver +
  ignition gate; fire-as-ray-source (already deposits heat via `cast_fire_heat`); the shared-wind transport;
  ACES; the atmosphere/`wave_p` shockwave; decompression-clears-smoke; the fire own-tile plume pressure deposit
  (`06` §5 stage 2 — a *precedent* for the expansion tap).

---

## 4. Beautiful-rendering roadmap (prioritized, status-marked)

Tiered by beauty-per-effort. **[SPECCED]** = design exists in canon, unbuilt. **[PARTLY]** = half-shipped.
**[NEW]** = not in canon. Render-only items are float and feel-gated (your eye); sim items change numbers and
need golden re-baselining.

### Tier A — cheap, mostly wiring (the high-payoff core)
1. **Black-body emission as the ray E-A source term** — **[SPECCED]** (§6.2 sub-note + §6.1 step 4). The
   absorption + additive-scatter compositing it folds into already ships. *#1 payoff per line.* Turns "coloured
   fog with light passing through" into actual glowing fire/embers/soot. ~1 LUT fetch + 3 MADs/emitting tile.
2. **`glow_temperature` proxy field (§2 option b)** — **[NEW, but mandated by canon]**: the only
   architecturally-legal way to feed the §6.2 emitter the cooling-plume memory. +1 advected float slice (batches
   into the existing transport) + a T⁴ decay.
3. **Curl-noise (Bridson) divergence-free detail velocity** — **[SPECCED]** (§6.1 step 2a names it explicitly).
   **Do NOT build vorticity confinement:** Breach's wind = `−grad(potential)` is mathematically **curl-free**,
   so confinement has nothing to amplify. Curl-noise is the missing rotational component that makes smoke roll —
   both as a sub-tile swirl nudged into advection and as the wisp-velocity for the §6.1 normal/flow-map.
4. **Combustion three-tap pipeline (heat + soot + EXPANSION→pressure)** — **[PARTLY/SPECCED]**: the fire field,
   `fuel_gas` flammable flag, and pressure-as-O2 gate exist; §6.2 specs the flammable consumption + soot/steam
   products as unbuilt. The **expansion→pressure tap** is the genuinely-new high-value wire — it makes a
   fireball *billow/pop* outward instead of just glowing. (Precedent: fire already does an own-tile plume
   pressure deposit, `06` §5 stage 2.)
5. **Explosion = transient {pressure impulse + glow flash + soot dump}** — **[PARTLY]**: `apply_explosion`
   already clears inner-40 % smoke, sets `fire` on flammable tiles, and `add_explosion_smoke` deposits; `wave_p`
   already pushes units. New framing: a Friedlander overpressure shape `p(t)=p0+ΔP(1−t/τ)exp(−ατ/τ)` (the
   `(1−t/τ)` gives a free suction/inrush phase) + a bright `glow_temperature` flash through the emitter. Unifies
   fire + explosions + shockwave under one impulse — realises the floated "explosion tiles as transient ray
   light sources".
6. **Soot lifecycle: two-timescale split + O2-keyed oxidation sink** — **[PARTLY]**: per-gas `decay` is
   loaded-not-applied; *applying* it (plus an oxidation term keyed on pressure-as-O2) is the gameplay half.
   Sealed rooms go sooty; vented/breached rooms burn clean.

### Tier B — sim feel-gated (changes numbers, re-baseline goldens)
7. **Limited-MacCormack / BFECC anti-diffusion over the existing semi-Lagrangian advection** — **[NEW, sim]**.
   Highest beauty-per-FLOP sim upgrade because Breach already pays for SL; wraps it in 3 passes + a 4-neighbour
   min/max clamp to recover ~2nd order and stop wisps dissolving into mush. Pairs with curl-noise (curl adds the
   swirl, MacCormack stops it being smeared away). Deliberate behaviour change → feel-gate + re-baseline.

### Tier C — render shader build (zero sim cost, but a real pipeline)
8. **§6.1 render-half: per-pixel smoke NORMAL** (density-gradient silhouette + wind-tilt + advected curl/fbm
   wisp-normal + puffiness z) with **two-phase flow-map advection** — **[SPECCED]** (§6.1 steps 1–2, fully
   worked out; the normal/wisp/flow-map half is the explicitly-unbuilt part — no shader path exists). In
   top-down the 2D smoke shape *is* the whole effect, so this is disproportionately valuable; it's what finally
   hides the coarse tile grid.
9. **Cheap self-shadow: normal·light_dir Lambert+wrap + 3-tap Beer–Lambert bulk shadow** — **[SPECCED]**
   (§6.1 step 3). Lit-face/dark-face/rim cues = genuine volume from above.
10. **Single-scatter in-scatter polish (the `smoke_glow` god-ray path)** — **[PARTLY]**: the `smoke_glow` buffer
    + decoupled per-gas `scatter_albedo` already ship; refine to a proper single-scatter term (optional mild
    Henyey–Greenstein), make steam catch light brightly.
11. **PBR mip-pyramid bloom (Karis-average) + per-channel ACES verification** — **[PARTLY]**: ACES ships
    (verify per-channel vs luma-only); bloom may need the CoD:AW pyramid. The polish that wraps HDR fire cores
    in glow; only pays off once item 1 produces HDR cores.

### Tier D — big lift, later (after the cheap wins + CUDA water S3)
12. **Fire as a real flickering ray light source into the scene** — **[SPECCED]** (`06` §"Fire as a light
    source"; `08` deferred "hot-tile emission"). Fire's *heat* cast already runs in-sim (`cast_fire_heat`); the
    *render light* cast is a planned mechanical relocation into the sim. Schedule onto the CUDA raycaster, built
    for many-source parallelism. Flicker = fixed-seed noise keyed to (tile, tick), never `rand()`.
13. **Sparse / active-tile simulation** — **[NEW, enabling]**: 8×8 block active-list (active if it holds
    smoke/fire/velocity above a fixed threshold or is adjacent to one). Must be bit-identical to dense (fixed
    threshold + fixed block order). The budget substrate for item 14.
14. **Multi-layer 2.5D smoke (K stacked z-layers sharing one heightmap + 1D conservative vertical buoyancy
    exchange)** — **[NEW; "one of the biggest"]**. Architecturally it's the existing batched 2D solver × K
    (`gmap.gas[gas, z, y, x]`), reusing every kernel, plus a thin per-column ±1-in-z conservative transfer (hot
    gas goes UP — the physically-correct top-down reinterpretation of buoyancy, and a real home for the plume
    "rise + cool" arc). The whole payoff is the ray engine: today every tile is a single optical depth; with K
    layers the per-channel march sums optical depth through real vertical structure, upper layers self-shadow
    lower ones, glow/god-rays gain vertical falloff, the shared heightmap clips layers against geometry. This is
    where the top-down sim stops looking 2D. Gated behind the cheap wins, sparse-tile sim (cost is ×K), and the
    CUDA water solver (S3) — start K=2 to validate the transport, grow to ~8–10.

### Related floated ideas (do not re-invent)
- **Smoke difference field** (difference of two perturbed advections) — **[NEW]**: a near-free FTLE-flavoured
  turbulence *mask* to modulate where wisps/emissive-edges belong. Curl-noise (item 3) is the principled wisp
  *generator*; the difference field is a refinement *modulator* on item 8. Park it there.
- Baked flipbooks / Valve flow-maps for **decorative/distant** pyro (set-dressing only); screen-space radial
  god-rays for **hero moments only** (explosion flash, the breach) — both low-priority polish.

---

## 5. Steam rename + the water interface (FLAG, do not design)

**Rename:** `white_smoke → steam` is a pure rename — the §6.2 table already labels the row "water vapour /
steam" and notes its visual is `scatter_albedo`-dominated near-white brightening, **not** absorption (do not
raise its absorption). Action: rename the `[gases.white_smoke]` config key + the GasTable id + `gmap`
references + the readable-summary table; optics values unchanged; no behaviour change.

**Water interface — owned by the separate water effort (Fable); do NOT touch `07_fluid_and_water.md` or design
the phase change.** Three couplings the combustion pipeline will *want* but must not implement here:
1. **Source:** combustion's `vapor_yield` is steam (combustion is a steam source) — Breach's side only emits.
2. **Boil:** hot water tile → emits steam (a steam source the water solver owns).
3. **Condense + cool:** steam on a cold/wet tile → water (a steam sink), and vaporization **cools its tile**
   (couples to `temperature` — exactly the water↔fire interaction Erik flagged in `06`'s comments: water has 3
   states in pressure, gas+solid in vacuum; vaporization cools).

Clean interface to hand Fable: **combustion writes steam density (a source); the water solver owns
boil/condense/vaporization-cooling as steam↔water mass+heat exchange.** Surface as a named dependency, not a
spec.

---

## 6. Determinism + the two render-pipeline preconditions

Everything new in the lifecycle is **float / render-only** and crosses **no gameplay threshold** → no
fixed-point, no lockstep/ML risk — *provided*:
1. `glow_temperature` is never read by ignition/damage/AI;
2. soot re-emits **light only**, not heat, in v1 (see below);
3. the combustion **gate** keeps using the existing fixed-point `temperature`;
4. the LUT is a checked-in float64-baked constant table (no per-frame platform `exp()` divergence);
5. any flicker is a pure function of (tile, tick), never `rand()`.

**Soot re-emits LIGHT (yes — free, safe).** A hot soot/fireball tile as a ray light source lights nearby
walls/units/smoke through the existing per-channel raycaster, and firelight is correctly reddened through
intervening soot (red survives, blue dies first) for free. `light_rgb` is float/render-only. Cap active emitter
count per tick (brightest-K / cluster) to bound the ray budget; the CUDA raycaster is built for this.

**Soot re-emits HEAT (recommend OFF in v1).** If hot soot also deposits into the Q16.16 `heat` buffer you get a
feedback loop (heat → glow_T → emission → heat) that can run away — and worse, it is a **gameplay** path (heat
ignites/damages) that must stay bit-identical on the determinism-critical decoupled heat channel
(`08` §"Determinism", `--fmad=false`, host-precomputed dirs, double-precision quantize). Keep the light/glow
path free to use float `exp/cos/sin`; keep the heat channel pristine. If soot-reheats-soot is wanted later,
drive it from a fixed-point proxy and gate it behind the heat bit-identity test.

**Two preconditions to verify before any tuning (checks, not design):**
- (a) **ACES is applied per-channel, not luminance-only** — luma-only will NOT roll fire cores to white
  (Narkowicz's own caveat). `shaders/lighting.fs` applies an ACES filmic tonemap; confirm the form.
- (b) **Emission stays UNCLAMPED HDR until the tonemap** — else the T⁴ cores clip to flat saturated patches.

**Curl-noise determinism:** pin a fixed integer hash under `/fp:strict`, OR (cheaper + trivially bit-identical
CPU/GPU) sample a host-precomputed tiled noise texture. Lean toward the precomputed texture given the
lockstep/ML + CUDA bit-identity mandate.

**Sim-side writes (MacCormack, expansion tap, Friedlander impulse)** route through the existing fixed-point
`wave_p`/atmosphere quantization boundary and are **feel-gated behaviour changes** (not bit-identical) → require
golden re-baselining.

---

## 7. Sequencing against the post-CUDA roadmap

Your stated post-CUDA order is **weapons → units → NN training → graphics overhaul**. Graphics is a **parallel
track**, not a replacement for that line — but it should *lean on the physics* so the two reinforce each other:

- **Weapons** (energy-weapon pre-phase, `08`) directly *exercises* the emission + explosion work: a laser
  breach lighting the room it just opened, a muzzle flash through smoke. Build the **black-body emitter (items
  1+2)** early because weapons want it anyway — it is shared infrastructure, not pure cosmetics.
- **Units** want the **soot-as-light + explosion impulse** (units pushed by shockwaves already exists) and the
  O2-keyed oxidation aftermath for readable tactical state. Light coupling only; no new gameplay determinism.
- **NN training** runs **headless** — it consumes only `heat`/`temperature`/gas densities, none of which the
  render track touches. So Tiers A–C are *invisible to training* and can proceed in parallel without risk,
  **provided** the `glow_temperature` discipline holds (never a sim input) and soot-heat stays off. The one
  sim-side item that *does* touch training trajectories is **MacCormack (item 7)** and the **expansion tap /
  Friedlander impulse** (items 4–5) — those change the physics the NN sees, so land them *before* a training
  run is blessed, not during, and re-baseline goldens.
- **Graphics overhaul** is then mostly the Tier-C shader build (items 8–11) + Tier-D (12–14) on the matured
  CUDA raycaster + after CUDA water (S3).

**Suggested interleave:**
- **Now / alongside weapons:** Tier A items 1–2 (emitter + `glow_temperature`), then 3 (curl-noise), then 4–6
  (combustion three-tap + explosion impulse + soot lifecycle). Items 1, 2, 3, 5(glow half), 6 are pure
  render/feel-gated; item 4's gate + item 5's impulse touch the fixed-point boundary.
- **Before any blessed NN run:** Tier B item 7 (MacCormack) + finalise the expansion/Friedlander numbers, then
  re-baseline.
- **Graphics-overhaul phase:** Tier C (8–11), then Tier D (12 fire-as-light on CUDA, 13 sparse-tile, 14
  multi-layer 2.5D) after CUDA water S3.

---

## 8. Decisions for Erik

1. **The crux:** approve the render-only advected **`glow_temperature` field (option b)** as the warmth-carrier
   (vs (a) no-new-field heat-deposit-only). Strong recommendation: **(b) as spine + (c)'s transient boost
   optional**. Confirm you accept ONE extra advected float field with the **iron rule that it never feeds
   gameplay** (so the no-air-temperature foundation stays intact). (a)-alone gives fireball-on/off, no cooling
   plume.
   **→ STATUS 2026-07-05: deliberately deferred** pending the EOS research pass (`docs/eos_research_brief.md`,
   roadmap Phase 1). If the ideal-gas reframe (§0) is adopted, (b) is moot — the real gas `T` carries the warmth
   and feeds the same LUT. (b) remains the fallback spine if the EOS is deferred or rejected.

2. **Soot re-emits LIGHT yes (free; realises "explosions as light sources"); re-emits HEAT no in v1** (avoids a
   runaway feedback loop on the determinism-critical heat channel). Agree to defer soot-reheats-soot behind the
   heat bit-identity gate.
   **→ DECIDED 2026-07-05 (Erik — independently re-derived twice, then confirmed):** heat is a strict one-way
   channel. Gas/soot **absorbs** heat (into its glow state) and **never re-radiates** it; heat ray-sources remain
   combustion/weapon events only. A third independent rationale joined the two above: the **ray budget** — heat
   only acts through rays, so every-hot-tile-as-heat-source would cost O(plume area) ray casts vs O(burning
   tiles) with absorb-only. Light is unaffected because it is two-tier: per-tile in-march glow is ray-free
   (the whole plume glows for ~3 MADs/tile), and only the brightest-K tiles are promoted to actual ray-casting
   `LightSource`s. Honesty note: real soot *does* re-radiate (Kirchhoff — firestorm preheating); we deliberately
   drop that effect since fire spread already has its own radiation mechanism and re-radiation would double-dip.
   The escape hatch (fixed-point proxy gated behind the heat bit-identity test, now proven via `cuda-breached`)
   stands if it is ever wanted.

3. **The expansion→pressure tap** (burning/detonating tile injects positive `wave_p`/atmosphere → the fireball
   self-pushes/billows). It is the difference between "glows in place" and "pops outward", and it has precedent
   (fire's own-tile plume deposit). It touches the fixed-point atmosphere/`wave_p` solver (the only piece here
   that touches gameplay determinism) → feel-gated behaviour change, re-baseline goldens. **v1 or fast-follow?**

4. **Vorticity confinement is OUT, curl-noise is IN** (wind is curl-free, so confinement has nothing to
   amplify; §6.1 already names curl-noise). Confirm dropping confinement entirely. And the **curl-noise
   determinism choice**: pinned hash under `/fp:strict` vs **host-precomputed noise texture** (recommended for
   bit-identity).

5. **`white_smoke → steam` rename** (approve — pure rename). The water↔steam phase interface
   (boil/condense/vaporization-cooling) is **handed to Fable as a named dependency** — combustion only emits
   steam as a source; we do NOT design the phase change. Confirm flag-not-spec.

6. **MacCormack anti-diffusion (item 7)** is a deliberate sim behaviour change (sharper smoke, different
   numbers) → feel-gated, golden re-baseline. Confirm you want it, and that it lands **before** the §6.1 render
   half (sharper density makes the normal/wisp layer look better) and **before** any blessed NN run.

7. **Multi-layer 2.5D smoke (item 14)** gated behind (a) the cheap wins, (b) sparse-tile sim (cost ×K), and (c)
   **after** CUDA water S3 (not competing with it). Start **K=2** to validate the inter-layer transport, grow to
   ~8–10. Confirm.

8. **Two render preconditions** (checks, not design): verify ACES is **per-channel** (not luma-only — else fire
   cores won't whiten) and emission stays **unclamped HDR** until the tonemap. If ACES is luma-only, approve the
   per-channel switch (or Stephen Hill's fit) as part of the emitter work.

9. **Build order / what's v1.** Proposed v1 = items 1, 2, 3, 4, 5, 6 (Tier A). Confirm the order and which steps
   are v1 vs fast-follow.

10. **This stays a BRAINSTORM.** When you pick the first slice I write it up as a proper `engine/05` + `engine/08`
    chapter update (Depends-on header + impl-status), per the design-docs-are-canon rule.
