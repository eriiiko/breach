# RULING — radiation law, raycaster scope, extinction shape (Fable design pass, 2026-07-31)

Answers `docs/radiation_and_raycaster_design_seed_2026-07-31.md` (Q1, Q2, §3, §4).
Reading order: seed → this ruling → the patch spec (§2 below). Opus executes from
this doc; nothing here is built yet. Numbers are marked **[measured]** (a bench ran)
or **[derived]** (algebra on verified relations — treat per the seed's §6 method
note: two derived floors in this arc were wrong before they were measured).

**The one rule everything below follows from:**

> **Heat moves between tiles only as an accounted exchange against a potential —
> conduction against `T`, radiation against `E(T) = ε·σ·T⁴` — antisymmetric by
> construction: what one end loses the other gains, exactly. Energy enters the
> world only at combustion sites and leaves only through declared sinks. There is
> no one-way painting of heat.**

Today's `k_fire_heat` ray deposit violates this rule twice: it deposits the same
per-ray energy at *every* marched cell (a painter, not a transport — a 5-cell ray
deposits 5× its payload), and nothing is ever debited from the source. Both the
far-field wall (§5.3) and the divergence hazard Erik caught (seed §1.2) are
consequences. The rule kills both by construction.

---

## 0. Erik rulings recorded 2026-07-31 (this ruling builds on them)

- **R-a — linear capacity response.** The linear-in-O₂ equilibrium (Peatross–Beyler's
  own shape) supersedes the curved enrichment anchors (0.49 @ X=0.25 / 0.67 @ X=0.30).
  New anchor row: §1 A3.
- **R-b — the capacity law is §4's answer.** The seed §4.0 constant-mortality sketch
  is superseded: constant mortality alone does not decouple (it yields `I_eq = 1 − r/a`,
  death at `a = r` — the same one-ratio-two-jobs trap). The decoupling that works is
  moving the fire's size into the logistic's *carrying capacity* (A3).
- **(Standing, from the seed)** §3 plume→T shim: REMOVED, not fixed. §5 sequencing:
  radiation is decided first; the raycaster scope follows it.

---

## 1. The five answers

### A1 — Q1: YES. Heat transport becomes a net-T⁴ radiation exchange. [GO]

For every (emitter `s`, absorbing marched cell `r`) pair the ray march already
enumerates, with per-ray share `w = 1/ray_count` and running material transmittance
`τ` (today's `heat_survival`):

```
net      = τ · w · ( E[T_s] − E[T_r] )        // signed, Q16.16 heat counts/tick
rad[r]  += net                                 // receiver gains
rad[s]  −= net                                 // emitter loses THE SAME integer
```

**Why this exact shape, point by point:**

1. **Antisymmetry is load-bearing, not preferred** (Erik's catch, seed §1.2). The
   same truncated integer `net` is applied `+` to one end and `−` to the other —
   the fixed-point kit's S1 conservation idiom (`fixed_point.h` mul-wide note:
   identical truncated value cancels exactly). Two equal-temperature fires read the
   same table entry, `net == 0` **exactly**, and divergence is impossible by
   construction — not by tuning. This is gate (e) in §3.
2. **The self-radiation question answers itself.** The source tile is its own first
   marched cell: `E[T_s] − E[T_s] = 0`. The seed §3 ruling ("our radiation will not
   heat its own tile") falls out structurally; the existing source-tile
   self-occlusion *skip* (`raycaster.cpp:287-305`) stays, so rays still leave the
   emitter unattenuated.
3. **`E(T)` is a baked integer table — mandatory, not an optimisation.** T⁴ at game
   temperatures overflows Q16.16 by ~10⁸ (seed §1.3). Bake at load, in double, then
   quantize (the locked S1 boundary idiom): index `t = clamp(T, 0, T_MAX_PHYS) >> 2`
   (4-game-unit buckets, 4000 int32 entries, ~16 KB — CPU array / CUDA `__constant__`
   or global), value
   `E[t] = quantize_heat( scale · (293 + 2·T_mid(t))⁴ )` with σ, the 0.833 m² face,
   the per-tick dt and the game/Kelvin mapping all folded into `scale` at bake time.
   No interpolation: a staircase in 4-game steps means near-equal pairs land in the
   same bucket and net *exactly* 0 (helps point 1); the step error at 1000 K is a
   few percent of E — below the limiter's granularity. **[derived; bake-time choice]**
4. **Emissivity = absorptivity = the existing `heat_atten` column (Kirchhoff).**
   The material that blocks heat rays is the material that emits/absorbs them —
   physically Kirchhoff's law, practically **zero new per-material dials** and the
   deposit gate stays the existing `heat_survival > heat_cull` gate, so the
   heat-touched tile set keeps its determinism contract (material-only, no gas
   `exp`). Consequence: **air (heat_atten 0) neither absorbs nor receives** — the
   painter's air-heating dies with the painter. See §5.3 for what that does to the
   far-field wall, and E3 for the room-feel escalation.
5. **Absorption stays lumped** (seed §1.4): the deposit converts through each end's
   own `heat_inv_shift`, i.e. the existing object branch
   (`temperature_solver.cpp:263-269`, same idiom as combustion's object site,
   `combustion.cpp:288-293`). Equal *energy*, different *ΔT* per material —
   physically right, already built.
6. **The stability bound is a flux limiter, designed in** (seed §1.5.2). Per pair,
   before applying:
   `budget_end = (|T_s − T_r| << heat_inv_shift_end) >> LIM_SHIFT` for each end;
   `|net| ≤ min(budget_s, budget_r)`, `LIM_SHIFT = 4`. Each ray may close at most
   1/16 of the pair's gap per tick per end; 8 rays → aggregate ≤ 1/2 — strictly
   inside the convex-stability line conduction sits on (4 faces × 1/4 = 1). Shifts
   and compares only. Because every transfer closes a fraction of a *gap*, the
   exchange is maximum-principle-shaped: no overshoot, no undershoot below the
   colder end. Citation to archive with the patch: Levermore & Pomraning 1981
   (flux-limited diffusion) + Howell/Mengüç/Siegel, *Thermal Radiation Heat
   Transfer* (net exchange, view factors) → `docs/papers/`.
7. **Signed accumulation needs its own plane and a signed conversion.** Two traps,
   both found by reading, both spec'd here:
   - Today's `heat[]` contract is *positive saturating* adds (order-free only
     because positives are monotone under saturation). Signed nets under saturation
     are order-*dependent*. Radiation therefore accumulates into a separate
     `rad_net[]` int32 plane with **plain (non-saturating) adds** — order-free for
     signed integers — with a static bound: |per-pair| ≤ limiter ≤ 2¹⁴ counts,
     ≤ 8 rays × ≤ 64 pairs/cell ⇒ |Σ| < 2²³ « 2³¹. CUDA: plain `atomicAdd(int)`.
   - The heat→T conversion **skips non-positive deposits**
     (`temperature_solver.cpp:259` `if (deposit <= 0) continue`) — an emitter's
     radiative *loss* would silently never convert and fire would never cool by
     radiating. The `rad_net[]` fold is a **signed** conversion:
     `dT = shr_round0(rad_net[i], heat_inv_shift[i])` (symmetric round-toward-0 so
     +x/−x behave identically), saturating add into `temperature[i]`, gas cells via
     the existing absorption-proportional branch. `heat[]` and its painter contract
     are left untouched until P-R4 deletes the painter.
8. **What still emits.** Emitters = burning tiles ∪ `thermal_solid` tiles with
   `T ≥ T_emit_gate` (new global; propose 60 game = 413 K, where εσT⁴ ≈ 1.5 kW/m²
   ≈ 1% of flame flux — below it the exchange is COOL_SHIFT noise). A hot
   *non-burning* crate/steel slab now radiates to its neighbours and **loses** what
   it gives — post-fire char glow, warm walls behind steel, all fall out. NB
   `cool_shift` today lumps *all* losses including radiation; once radiation is
   explicit the blessed retune (P-R5) should expect furniture/wood `cool_shift` to
   drift up (slower non-radiative loss) — noted so nobody double-counts.

**Where the burning tile's own temperature now comes from — the load-bearing
consequence.** With the painter gone, a lone crate's radiation *nets to zero* at
the source and *loses* to cold surroundings; combustion must own the plateau. The
current combustion deposit cannot: **[measured this pass, from the code]**
`H_fuel · burn ≈ 4.2 heat-counts/tick` at the blessed operating point vs the
`k_fire_heat` painter's ≈ 19,000 — a factor ~4.5×10³. The fix is **not** to inflate
the Huggett-anchored gas constant. Combustion Pass A already computes each
claimant's demand share (`combustion.cpp:187-190` `dem[4]`); add a **per-claimant
fuel-bed deposit** — the flame heating its own fuel surface, which is real physics
(it is *how fires sustain*) and is not self-radiation:

```
heat[src_k] += (mul_q16(burn_k, H_BED_M) << H_BED_SHIFT)     // order-free positive add
```

with `H_bed = H_BED_M · 2^H_BED_SHIFT` split mantissa/shift because the required
magnitude (initial estimate ≈ 4.3×10⁵ heat-counts per unit N_O2 **[derived]**,
from `T*·2^(his−cs) / (burn_rate·dt·I·o2f)`) does not fit a q16 constant.
`H_bed` is a **calibrated lumped constant like `thermal_mass`** — Huggett-*shaped*
(∝ O₂ actually consumed) but not Huggett-*valued*, because `thermal_mass = 8`
already lumps the ~130× surface-layer factor (seed §1.4); say so in its comment.
Two physics wins fall out: the plateau now **sags with local O₂** (a choked fire
cools — backdraft-adjacent feel), and a fire that consumes nothing deposits
nothing (the design's "choked fire consumes nothing" made thermal).

The plateau algebra for the tune-loop header becomes **[derived — verify at gate]**:
`T* ≈ H_bed · burn_rate · dt · I · o2f_local · 2^(cool_shift − heat_inv_shift)`
(claim-structure factor ≈ 1 for the lone crate; measure, don't trust).

**Feasibility — the explicit deliverable (seed: "do not assume").** The march
itself is unchanged (same DDA, same survival, same tile set). Per *absorbing*
marched cell, radiation adds: 2 table loads, 1 subtract, 2 multiplies
(τ·w folded to one q16), 2 shifts + 2 compares (limiter), 2 atomic adds — vs the
painter's 1 multiply + 1 add. Air cells (the vast majority of marched cells) do
**less** than today: no deposit at all. Emitter count rises from `burning` to
`burning ∪ warm` — bounded ~2–4× post-burn by `T_emit_gate` **[assumption —
measured at gate (g)]**. Baseline: 600-source batched CUDA cast = 1.5 ms
**[measured, S8c]**; linear scaling ⇒ ~3–6 ms worst-case firestorm, sub-ms for
normal scenes. CPU reference path scales the same way in ray count. **Verdict:
fits the budget, with gate (g) enforcing it at 2× and E2 as the escape.**

### A2 — §3: the plume→T shim removal spec (ruled by Erik; spec'd here)

Delete `fire_simulation.cpp:251-311` (the own-tile plume energy deposit) and its
CUDA twin `cuda_fire.cu:239-259`. Retire `fire_pressure_gain`, `temp_gain_scale`,
`T_FLAME_MAX` (tombstone in config comments; the latter two exist only as C++
defaults — audit §5). The debug probe fields (`dbg_plume_dT`) go with it.
Expected effect **[measured, seed §3]**: `T*` drops a flat −6.3% (predicted/
measured agree to 3 decimals); on steel it would have been 19% — the shim was the
one `temperature[]` writer bypassing `heat_inv_shift`, and this closes P-EOS's
7th-writer violation. **No compensation retune at this patch** — `k_fire_heat`
dies two patches later anyway; the interim bench runs 6% cool, accepted.

### A3 — Q3/§4: the extinction shape — the CAPACITY law (Erik ruling R-b)

Replace the growth term's hardwired capacity `(1 − I)` with a resource-
proportional capacity `I_cap = c·a` — algebraically `k_grow·a·I·(1 − I/(c·a))`,
in which `a` cancels:

```
grow = k_grow · I · (avail·hot − I·INV_C) · (1 + k_wind_fan·W)     // INV_C = 1/c, load-baked q16
die  = k_die · (1 − avail·hot) · I  +  k_wind_strip · W · (1−I) · I   // UNCHANGED
```

Pinned left-fold order: `gap = avail·hot − mul_q16(I, INV_C)` (signed sub) →
`k_grow·I` → `·gap` → `·wind_fan`. `gap < 0` (fire above its capacity, e.g. O₂
just dropped) makes `grow` negative — decay toward the new capacity rides the
existing signed-delta path. One new config key `[physics.fire] I_cap_per_avail`
(= c), one retuned key `k_die`. `1/c ≈ 0.395` sits comfortably in q16.

**Why this is the decoupling** (and constant mortality was not): the sustain
threshold keeps its form `a > r/(1+r)` — but `r` no longer has to sit near the
operating point to hold the fire small, because *size* moved into `c`. Each dial
now has exactly one job:

| dial | its one job | value at the anchors [derived] |
|---|---|---|
| `I_cap_per_avail` (c) | fire size: `I_eq = c·(a − r(1−a)) ≈ c·a` | **2.53** → I_eq 0.210 at ambient/pristine/hot |
| `k_grow` | tempo: ramp e-fold ≈ 1/(k_grow·a) ≈ 3 s | keep **3.5**; Erik may slow it later — now it moves *only* tempo |
| `k_die` (r = k_die/k_grow) | where the death wall sits | **0.035** (r = 0.010) — the wall moves to the *physical* limits |

What the anchors become **[all derived — bench gates in §3]**:

| quantity | today (r = 0.080) | capacity law (r = 0.010) |
|---|---|---|
| headroom on `F·o2f·hot` | 1.242× ( = 1/(1−I_eq), the §4.0 identity) | **9.3×** — size and death decoupled |
| local X floor (F = hot = 1) | 0.1944 — `o2_frac_ext` 0.13 unreachable, dead code | **0.1386** — the Peatross–Beyler anchor is live again |
| hp floor (fuel) | 24.2/30 — 19.5% of the crate can ever burn | **3.2/30 — 89% burns; fuel-governed death exists** |
| `h_min` → T_sustain → bootstrap | 0.806 → 212.2 → ratio 0.478 | **0.108 → 184.3 → ratio 0.416** |
| I_eq(X): 0.21 / 0.25 / 0.30 / 1.0 | 0.21 / 0.50 / 0.67 / 1.0 (hyperbolic, knife-edged near ambient) | **0.21 / 0.33 / 0.47 / 1.0 — linear (Erik ruling R-a)** |

Q16.16 quantum check (the `95bdec0` trap): net at the seed ≈ 40 counts/tick at
k_grow 3.5 **[derived]** — an order of magnitude above the truncation floor; the
general condition `dt·k_grow·seed·(a − seed/c)·65536 ≥ 2` goes in the tune-loop
header. `I_min = 0.02` stays (numerical ember floor). `wall_damage` gets re-sized
at P-R5 for the 6–8 min *fuel-governed* death that is now reachable — ballpark
`duration ≈ hp_burnable/(wall_damage·Ī)` with Ī ≈ 0.12–0.15 ⇒ wall_damage ≈
0.45–0.65 **[derived, ballpark only]**.

**Ride-along (seed §4.4, audit §1.2 — same lines, same patch):** `fire_T_ext`
becomes per-material by derivation, `fire_T_ext[mat] = ignition_temp[mat] − Δ`
with ONE new global `ignition_to_ext_delta = 100` — furniture: 280 − 100 = **180,
exactly the blessed value** (zero feel change on the bench crate); wood: 300 −
100 = 200. The invariant `fire_T_ext < ignition_temp` becomes structural.
`fire_T_span` stays global (40). `ignition_seed` stays an explicit dial but gains
a **load-time check** per flammable material: `seed ≥ 1.15 · I_sustain[mat]` →
console warning naming the material (full auto-derivation deferred; audit §1.4).

### A4 — Q2: the raycaster

1. **The source build moves into C++**: `cast_from_fire_plane(fire, mats, dials)`
   (+ CUDA twin in `cuda_raycaster.cu`) replaces the Python per-tile
   `bp.LightSource()` loop (`physics_runner.py:1097-1130`, ~6000 pybind attribute
   writes/tick at 600 fires). Every per-source parameter is already a pure
   function of `(x, y, I)` — including the deterministic `angle_center` hash and
   the per-ray `1/ray_count` heat split (`raycaster.cpp:540,589`) — so the C++
   build reproduces the list exactly: row-major order, jitter 0, same floats.
   **Gate: `heat` byte-identical** (this patch changes no law — P-R1).
2. **`Raycaster::update_from_fire` and `coarse_cluster` are deleted**, not
   revived: no production caller (audit §7c), legacy intensity-only signature,
   an RNG jitter land-mine, and clustering is *incompatible with the radiation
   law anyway* — a merged pseudo-source has no well-defined `T_s`, and the net
   form needs the real emitter temperature. The cost discipline stays "many
   sources × few short rays" + `T_emit_gate`.
3. **The 8-ray fan and the range model survive this arc unchanged.** Under P-R4
   the same builder takes the emitter mask (burning ∪ warm) and per-tile `T_s`
   instead of a scalar heat payload; the fan geometry, survival machinery and
   deposit gate are untouched. Low-hanging fruit beyond that (shared render/sim
   casts, adaptive fans) is explicitly deferred — nothing here blocks it.

### A5 — what survives: the heat-currency census (the method lesson, applied)

After the arc there are exactly **two potentials and three energy writers**, and
the gate enumerates them (no grepping near keywords):

- Potentials: `T` (conduction, `temperature_solver.cpp` Pass 2) and `E(T)`
  (radiation, the new pass). Both antisymmetric pairwise exchanges.
- Writers into heat planes: **combustion** (`H_fuel` gas-side at the air cell,
  `H_bed` claimant-side at the fuel bed), **radiation** (`rad_net[]`, signed),
  and **weapons/payloads** (unchanged external deposits).
- `temperature[]` writers: TemperatureSolver ONLY (convert / conduct / cool).
  The shim deletion (A2) closes the one violation. `k_fire_heat`,
  `fire_pressure_gain`, `temp_gain_scale`, `T_FLAME_MAX` no longer exist.

---

## 2. The patch sequence (each lands alone, gated; Opus executes)

| # | patch | law change? | gate |
|---|---|---|---|
| P-R1 | raycaster source build → C++ (`cast_from_fire_plane`), delete `update_from_fire` + `coarse_cluster` | none | (a) `heat` **byte-identical** CPU & CUDA vs main, 600-fire firestorm + playground; (b) suite failure-set == baseline 39/1817/5 |
| P-R2 | plume shim deletion (CPU + CUDA + 3 constants) | −6.3% T* | (c) measured `T*/(gain·I)` 1.063 → **1.000 ± 0.01** on the bench burn; (d) CPU↔CUDA tol 0 step+resident |
| P-R3 | capacity law + per-material `fire_T_ext` derivation + seed load-check | yes (I-dynamics) | (e′) bench: I_eq 0.210 ± 5%, X-floor probe dies at X_local 0.139 ± 0.010, wall_damage-0.55 probe reaches hp < 5 (vs 25.5 today — the sign flips), enrichment row X=0.25 → I_eq 0.33 ± 0.03; (d) tol 0; quantum check ≥ 2 counts/tick |
| P-R4 | radiation law: E-table, `rad_net[]` plane, signed fold, limiter, `H_bed`, `T_emit_gate`, painter + `k_fire_heat` deleted | yes (heat transport) | (e) two-adjacent-equal-fires: `rad_net` between them **exactly 0**, unequal pair converges monotonically, no `T_MAX_PHYS` hit; (f) lone-crate plateau in [400, 500] via `H_bed` with I_eq 0.21; (g) 600-emitter batched cast ≤ **2×** the 1.5 ms baseline; (h) neighbour-crate ignition across a 1-tile air gap occurs (the seed §1.1 sanity number made live); (d) tol 0; far-field rise recorded (no target yet — E3 owns feel) |
| P-R5 | blessed joint tune (Erik + the loop), **new fuel-bearing golden created at the blessed dials** (bench §8 item 26 — today no golden can see a fire law), canon fold into `architecture/engine/06`, archive the seed + this ruling's brainstorm ancestry | dials only | HUMAN-TEST (feel-adjacent — nothing above auto-merges into main), the arc's ONE deliberate rebase stays unspent unless a legacy golden moves (none should: no committed golden contains fire) |

Worktree discipline per the master file: one git-touching agent per tree; design
docs committed to the branch before any dependent agent spawns.

## 3. Escalation triggers (stop and bring it back to Erik)

- **E1** — gate (f) fails with `H_bed` inside a defensible lumped range (±3× the
  4.3×10⁵ estimate): the combustion-owns-the-plateau premise is wrong somewhere;
  fallback named in advance = a small honest `k_bed_boost` dial, but that is a
  *ruling change*, not a tuning fix.
- **E2** — gate (g) fails (cost > 2× baseline): deterministic scope-reduction
  options only (raise `T_emit_gate`, shorten warm-emitter range) — never a
  nondeterministic emitter cap. If still failing, Q1 scope returns to Erik.
- **E3** — room feel: with the painter dead, far-field air warms only via
  combustion gas deposits + plume transport, and `c_v`/gas-side `H_fuel` are
  explicitly unanchored (audit §7b). If the room reads dead-cold to Erik at
  HUMAN-TEST, that is a *calibration mini-pass* on the gas side, not a licence to
  resurrect the painter.
- **E4** — P-R3 bench floors off by more than 2× their derived values: stop, re-
  derive; the seed's method note stands (two derived floors have been wrong).
- **E5** — any P-R4 divergence or `T_MAX_PHYS` hit in the two-fire gate: stop.
  That gate is the whole point of the net form.

## 4. The dial ledger (what dies, what is born, what retunes)

- **Dies:** `k_fire_heat`, `fire_pressure_gain`, `temp_gain_scale`,
  `T_FLAME_MAX`, `coarse_cluster`, `update_from_fire`, `fuel_ref` (already
  inert), the painter deposit itself.
- **Born:** `I_cap_per_avail` (c = 2.53), `H_BED_M`/`H_BED_SHIFT` (lumped, one
  logical constant), `T_emit_gate` (60), `ignition_to_ext_delta` (100),
  `LIM_SHIFT` (4, a stability constant not a dial), the `E(T)` bake (ε from
  `heat_atten` — zero new per-material columns).
- **Retunes at P-R5:** `k_die` → 0.035, `wall_damage` → ~0.5 (fuel-governed
  6–8 min), `cool_shift` may drift up (radiation now explicit), `k_grow` only if
  Erik wants a slower ramp — it is finally free to move alone.
- **Unchanged anchors:** `burn_rate` 0.02 (Huggett), `fuel_per_o2` 0.7,
  `o2_frac_ext` 0.13 (Peatross–Beyler — *and it means something again*),
  `o2_frac_full` 1.0, `H_fuel` gas-side 4.0, `thermal_mass`, `heat_atten`,
  `ignition_temp`.

## 5. Measured evidence this ruling stands on (2026-07-31 session)

1. **Probe A** — `wall_damage` 0.083 → 0.55, all else blessed
   (`_fire_tuning_artifacts/probe_walldamage_055.csv`): death 334 s → **74.8 s**,
   final hp **25.53**/30 (floor 24.2 respected; naive "burns to zero" falsified).
   §4.0's Consequence B is measured fact: today no dial makes a crate burn down.
2. **Probe B** — `k_die` 0.28 → 0.315 (r 0.080 → 0.090), all else blessed
   (`_fire_tuning_artifacts/probe_kdie_0315.csv`): peak I **0.126** (never left
   the seed), max T 280 (never grew), dead at **59.5 s**, hp 29.53. A +12.5% move
   of one ramp dial removed the fire's ability to exist at ambient — the
   one-margin coupling, demonstrated in one run. (Prediction ledger: direction
   right; my isolated-floor death-time estimate was 4× slow because the floors
   compound through the T feedback — the viable region is *tighter* than §4.0's
   per-factor table.)
3. **The identity** behind both: in the current law, headroom on `F·o2f·hot` is
   `1/(1−I_eq)` — the §4.0 floors (0.805/0.806/7.4%) *are* `1 − I_eq` plus an
   O(a) correction. Asking for a small fire is asking for a nearly-dead fire.
   The capacity law severs exactly this identity.
4. **Painter arithmetic** [from code, this pass]: combustion's current object-site
   deposit ≈ 4.2 counts/tick vs the painter's ≈ 19,000 at the blessed plateau —
   the factor `H_bed` must carry, and why A1 spends a section on it.

## 6. Standing constraints (inherited, restated so the patch agents see them)

Q16.16 only in the sim path, no libm (`test_no_float_in_sim_tu`); every patch
gates CPU↔CUDA at tol 0, step and resident; suite failure-set must equal the
inherited 39/1817/5; byte-identity wherever the law is unchanged; ONE deliberate
golden rebase for the whole arc, unspent until P-R5 and only if actually needed;
feel-adjacent ⇒ HUMAN-TEST before merge; source order row-major, deposits
order-free (positive-saturating for `heat[]`, plain-signed for `rad_net[]` — the
contracts differ and the comment at each site says why); credit-the-source
headers + `docs/papers/` archives for Stefan–Boltzmann/flux-limiter references.

## 7. Deferred, explicitly (so silence is not read as a decision)

Per-material `k_grow`/`k_die`/`wall_damage`/`smoke_emission` (audit §1.5 — no
victim while both fuels are cellulosic); `soot_yield` per-claimant split (audit
§3 — structurally expensive); the `hp` structure/fuel conflation and the two
parallel soot sources (later arc); the object→gas convective term (EOS ruling
A2's Phase-3 fork — `k_wind_strip`'s replacement); render/sim cast sharing and
fan adaptivity (A4.3); gas-side `c_v`/`H_fuel` anchoring (E3 owns the trigger).
