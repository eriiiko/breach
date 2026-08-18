# P-F1b — THE RECALIBRATION AT PACKAGE A (2026-08-02)

**Status: BUILT AND GATED. Dials + calibration benches + config only — NO law
changes.** The patch that brings the fires back to life under the honest
radiation books P-F1a built. It runs `docs/fire_realism_design_2026-08-01.md`'s
F8 order (steps 3–6 in spirit) at THE SIZING RULING's package A: `DRAW_R = 2`,
`o2_potency = 1.0`, the measured R=2 supply, deep-orange plateau.

Feel-adjacent by construction: **HUMAN-TEST gate before merge.**

---

## 0. The one-line result

At P-F1a's frozen dials every bench fire died in under a second. It now:
ignites from its seed, ramps over ~20 s, holds a deep-orange plateau
(836–856 K), flickers, spreads to a neighbour across an air gap in ~30 s,
burns for 8 min (kindling) / 24 min (furniture), and dies **by the temperature
gate with 54–61 % of the crate consumed** — unless you seal the room, in which
case it **suffocates in ~2 minutes with the fuel almost untouched.**

---

## 1. THE DIAL TABLE (old → new)

| dial | old | new | ITS ONE JOB |
|---|---|---|---|
| `[physics.combustion] H_BED_M` | 25290.0 | **18125.0** | (with the shift) the fuel-bed deposit |
| `[physics.combustion] H_BED_SHIFT` | 3 | **4** | ⇒ `H_bed` 2.023e5 → **2.900e5**, χ_bed 0.540 → **0.7745** |
| `[physics.fire] k_grow` | 4.0 | **2.0** | TEMPO — the ramp, and nothing else |
| `[physics.fire] k_die` | 2.0 | **0.008** | WHERE THE DEATH WALL SITS (r 0.5 → **0.004**) |
| `[physics.fire] I_cap_per_avail` | 2.53 | **14.0** | SIZE — where the plateau intensity sits |
| `[physics.fire] ignition_seed` | 0.1 | **0.12** | clears the bootstrap floor with margin |
| `[physics.fire] ignition_to_ext_delta` | 100.0 | **200.0** | knee geometry — the `hot` ramp's FOOT |
| `[physics.fire] fire_T_span` | 150.0 | **180.0** | knee geometry — the `hot` ramp's WIDTH |
| `[physics.fire] wall_damage` | 0.4 | **0.03** | BURN DURATION (per family, via each row's `hp`) |
| `[physics.fire] T_emit_gate` | 180.0 | **310.0** | who CASTS — the gate-wall fix |
| `[materials.wood] cool_shift` | 5 | **13** | THE SPREAD DIAL (interim: the whole non-radiative residue) |
| `[materials.furniture] cool_shift` | 5 | **13** | ″ |
| `[materials.kindling] cool_shift` | 9 | **13** | ″ |

**Untouched, as ruled:** `burn_rate` (0.02, anchored), `o2_potency` (1.0),
`fuel_per_o2` (0.7), `o2_frac_ext` (0.13), `o2_frac_full` (1.0), every
`heat_atten` (feel), `draw_r` (2), every `thermal_mass`, `rad_scale` (DERIVED),
`k_wind_fan` / `k_wind_strip` (P-F3's anchors).

---

## 2. THE MEASURED PRIMITIVES (derive first, then measure)

Everything below is quoted on **ONE arena** — the P-F4a-chartered STILL-AIR
REFERENCE ARENA (`tools/fire_timing_harness.build_level`: 84×40 interior, tile
0.333 m, crate at (12, 21), planetside, sky-exchange refill on, `DRAW_R = 2`).
Arena size AND tile size both move the supply; an early solve on a 40×24 /
0.5 m arena was 55 % off and had to be redone. **Quote the arena or the number
is meaningless.**

`_fire_tuning_artifacts/` carries the raw CSVs; the probe is
`pf1b_probe1.py` in the session scratchpad (pinned-I, 40 s to settle).

### 2.1 The supply — `alloc(I)`, O2 units drawn per tick by one lone crate

| I | alloc [units/tick] | X_local | chemical kW | \|W\| (own plume) |
|---|---|---|---|---|
| 0.05 | 3.471e-5 | 0.2070 | 4.0 | 0.065 |
| 0.10 | 6.574e-5 | 0.2002 | 7.6 | 0.073 |
| 0.192 | 1.1749e-4 | 0.1950 | 13.6 | 0.072 |
| 0.30 | 1.5793e-4 | 0.1824 | 18.3 | 0.081 |
| 0.45 | 2.0866e-4 | 0.1747 | 24.2 | 0.096 |
| 0.60 | 2.6410e-4 | 0.1709 | 30.6 | 0.111 |
| 0.80 | 3.8439e-4 | 0.1716 | 44.6 | 0.102 |

Cross-validated **two independent ways** at every point, agreeing to ≤1.6 %:
(a) the law-agnostic "re-run the combustion pass on settled state and revert
every plane" draw probe; (b) the tile's own steady thermal balance (§2.2),
inverted. Two things fall out and both matter:

* **The fire starves its OWN ring.** X_local falls 0.207 → 0.172 as I goes
  0.05 → 0.80, i.e. `o2f` falls from 0.088 to 0.048 — roughly HALF the ambient
  0.092. The supply is genuinely the controlling channel (theorem T2), and the
  `o2f` the logistic reads is not the ambient one.
* **`alloc` is sub-linear in I** (≈ I^0.82), so doubling the fire buys less
  than double the oxygen.

### 2.2 The thermal balance (the measuring instrument, and the plateau law)

Per tick, game units, for one burning tile:

```
alloc(I) * H_bed / 2^his  ==  a * (E[T] - E[0]) / (2^his * 65536)  +  T / 2^cs
        deposit                    v7 rule-4 SKY term                cool_shift
```

with `E[T] = rad_scale * (297 + 8*floor(T/4))^4` — the baked table, exactly —
`a` = `heat_atten`, `his = log2(thermal_mass)`. A BURNING tile always casts, so
the sky term is unconditional for the burner (`T_emit_gate` does not enter).
Predictions match the engine to **≤1.6 %** at every row of §2.1.

**What this equation says, and it is the whole recalibration:** at the ruling's
deep-orange target a crate sheds **15.1 kW to sky** and **0.04 kW through
cool_shift**. RADIATION now owns >99 % of the plateau. `cool_shift` has stopped
being the plateau dial entirely — the P-R4-era derivation at the `H_bed` key
("radiation-out is ZERO and cool_shift is its only loss") is void, and is
superseded in place.

### 2.3 The self-plume wind is NOT negligible

`|W|` at the fire's own tile is 0.065–0.11 in *still air* — the plume's own
outflow. The logistic reads it twice: `(1 + k_wind_fan*W)` on growth and
`k_wind_strip*W*(1-I)*I` on death. At the operating point the STRIP term is
**0.030 s⁻¹ against `k_die`'s 0.008** — four times larger. A first solve that
ignored it mis-sized the death wall by a factor of four and produced a fire
that declined where the algebra said it should grow. Both benches now record
`|W|`, and the death-cause decomposition uses the full bracket.

---

## 3. THE DERIVATIONS, DIAL BY DIAL (derived → measured)

### 3.1 `k_die` — the death wall, set from physics not feel

Sustain needs `a = F·o2f·hot > r/(1+r)` with `r = k_die/k_grow`. Ambient air
gives `o2f = (0.21−0.13)/(1−0.13) = 0.0920`, so the shipped `r = 0.5` demanded
`a = 0.333` — **3.6× the maximum the atmosphere can supply**. No seed, no
temperature and no other dial could have saved it; the shipped pair had been
arithmetically fire-dead since the continuous-O2 law landed. (config's own note
at `o2_frac_ext` predicted exactly this: *"the split rescales the whole avail
axis by ~10x, so k_die/k_grow moves with it"*.)

`r = 0.004` puts the logistic's own extinction wall at

```
X_wall = X_ext + (X_full - X_ext)*r/(1+r) = 0.13 + 0.87*0.003984 = 0.1335
```

— just ABOVE `o2_frac_ext = 0.13`, so **the OXYGEN limit is the binding one and
`o2_frac_ext` is live code again.** P-R3's ruling named the opposite state
(*"o2_frac_ext = 0.13 was dead code, because the logistic wall bit first at
X = 0.1944"*) as the defect; this closes it. **Measured**: the sealed-room fires
die at X_local 0.111–0.122 — under X_ext, which is only reachable because the
logistic wall no longer bites first.

### 3.2 `H_bed` + `I_cap_per_avail` — the plateau, solved jointly

Two unknowns, two equations. §2.2 fixes `H_bed·alloc(I_eq)` from the target
temperature; the logistic's own equilibrium fixes `I_eq`:

```
I_eq/c = (1+r)*o2f_local(I_eq)*hot(I_eq) - r        (still-air, wind folded in)
```

Solved for T ≈ 280–300 game: `H_bed = 2.90e5`, `c = 14`. **χ_bed check**
(design F8 step 4, band [0.3, 1.0]):
`χ_bed = H_bed * 65536 * 1.968e-4 / 4.83 MJ = H_bed * 2.6706e-6 = ` **0.7745 ✓**.

**THE SPLIT HAD TO MOVE**: the mantissa must stay under 32768 (Q16.16 range),
so 2.90e5 needs shift 4, not 3. Deposit granularity is unchanged in kind (a
lone crate's per-tick claim is ~11 raw counts either way) and the int64→int32
deposit clamp fires on exactly the same full-drain cells it did before.

**AN HONEST CORRECTION TO THE RULING'S ARITHMETIC.** THE SIZING RULING says
"the watt books … are the measured R=2 supply (~11–13 kW lone-crate) ⇒ lone
still crate settles ~890 K". Those two do not connect at any honest χ_bed. A
crate at 890 K radiates **15.1 kW** (measurable directly: `rad_net` 3.198e6
counts/tick × 1.968e-4 J × 24). To supply 15.1 kW *into the bed* at χ_bed 0.77
needs ~19.5 kW of chemical power, i.e. I ≈ 0.45 — not the I = 0.192 at which
the 11–13 kW figure was measured. The 11–13 kW number was a measurement at a
PINNED operating point, not a ceiling; the fire is free to run harder and does.
**Landed**: kindling settles at 271.7 game (836 K) drawing ~19 kW, furniture at
281.7 (856 K). Both inside the ruling's 270–330 band; both below the 890 K
headline, and the gap is exactly the χ_bed < 1 the energy books require.

### 3.3 `ignition_to_ext_delta` + `fire_T_span` — the knee geometry

Together they place the `hot` ramp: foot at `ignition_temp − delta`, top at
`foot + span`. New feet/tops: kindling & furniture 80 → 260, wood 100 → 280.
They had to do two jobs at once:

* **The part-burn fraction.** Where the foot sits relative to the plateau IS the
  part-burn dial — a high foot kills the crate with most of its fuel intact
  (P-R3 measured 19.5 % burnt at Δ = 100), a low foot lets it burn down. Δ = 200
  measures **61.1 % (kindling) / 53.5 % (furniture)** — inside the 30–70 %
  interim band.
* **Wood has to be able to burn at all.** A solid `a = 1.0` wall sheds twice the
  sky flux of a permeable `a = 0.5` crate, so it plateaus ~240 game against the
  crate's ~310. At Δ = 150 wood's ramp foot sits so close to that plateau that
  wood has NO equilibrium — a lit wood wall dies. Δ = 200 restores it.

### 3.4 `cool_shift` 5/9 → 13 — the dial changed jobs

With radiation owning the plateau (§2.2) and `T_emit_gate` above every
`ignition_temp` (§3.5), a NOT-YET-BURNING neighbour does not cast at all — so
`cool_shift` is the only thing between an incoming radiative flux and that
neighbour's ignition. **It is now the SPREAD dial.**

A tile one air gap away is crossed by ~1 of the 8 fan rays, so it absorbs
`a_s·a_r·w·(E[T_s] − E[T_r])` per tick and sheds `T/2^cool_shift`. Requiring its
ceiling to clear `ignition_temp = 280` against a 300-game emitter:

```
2^cool_shift  >=  280 / 0.0643  =  4355     =>     cool_shift >= 13
```

**Measured** (`tools/fire_spread_bench.py`, one air gap, emitter pinned):
ignition in 66.0 s from a 300-game emitter, 54.5 s from 310, 39.2 s from 330;
at cool_shift 12 the same chain FAILS from a 300-game emitter (the receiver
tops out at 276). Derived floor 13, measured floor 13.

**STATED AS INTERIM.** 13 is an e-fold of 341 s, far slower than the ~7 an
honest natural-convection anchor gives (h ≈ 10 W/m²K on the design's 51.6 J/K
lump ⇒ 0.0067·T per tick ⇒ cool_shift ≈ 7.2). That gap IS the convective term:
design F8 step 3 hands `cool_shift` the whole non-radiative residue until F3
exists, and **the F3 patch re-runs this step**.

### 3.5 `T_emit_gate` 180 → 310 — THE GATE WALL

The sanctioned raise (Erik's recorded lean, *"increase the threshold
drastically, compensate by making radiation quicker"*; design v5.3 records it as
a tuning-session item with a stated accuracy cost).

**Why it had to move.** A receiver warmed by a fire crosses the gate, becomes an
emitter, and starts paying its OWN sky in the ~7 directions that leave the world
while still receiving on the ~1 that sees the fire. Its equilibrium above the
gate is therefore `E_r ≈ E_s/15`, i.e. `K_r ≈ K_s/1.97`: **a 893 K flame can
only hold a one-gap neighbour at ~450 K**, far under any ignition_temp. P-F1a
measured exactly this (a plank stalling at 183.7 game, four units above the old
gate of 180). Below the gate a solid does not cast at all, so its only loss is
`cool_shift` and it CAN be carried to ignition.

310 sits just above the highest flammable `ignition_temp` (wood 300), so no
flammable ever occupies a "hot but stalled" window — it ignites first, and a
BURNING tile is an emitter regardless of this gate.

**DEVIATION, NAMED.** The task order suggested "e.g. toward 250-280, below the
lowest ignition_temp". That range cannot work: gate (f) requires the wood chain
to reach wood's 300, and any gate at or under 300 stalls it below that. The
constraint that actually binds is *above the highest flammable ignition_temp*,
which is 300 → 310.

**The accuracy cost, stated.** A hot NON-flammable solid (hull/steel/glass/
doors) below 310 game (= 913 K) does not radiate its heat away. Bounded by
`E°(310) − E°(0)` per direction. It still ABSORBS normally (v7.3: a sub-gate
solid pays and receives via rule 1 whenever an emitter sees it — what it does
not do is CAST), so no energy is minted, only under-shed.

**The cost benefit, measured by P-F1a.** P-F1a's OPEN POLICY item: the caster
set is temperature-defined and unbounded, and its warm-halo case (every tile
within 3 of a fire above the gate) reached 19,424 emitters / 6.58 ms — 2.2× over
budget. At 310 the warm halo does not cast at all and the caster set collapses
to essentially the fires themselves. **This dial answers that open item; no new
cap mechanism was invented.**

### 3.6 `wall_damage` 0.4 → 0.03 — duration

Two drains share the fuel store: `wall_damage*I` per second, and combustion's
stoichiometric payment `fuel_per_o2 * alloc(I) * 24`. At the operating point
(I ≈ 0.26) they are **0.0078 and 0.0026 hp/s** — `wall_damage` still dominates
3:1 and is still THE duration dial. Solved for Erik's ruling-2 target ("a 30 kg
crate burning ~30 min is desirable").

**THE COUPLING, STATED AS OWED.** Fuel is still `hp`. There is no `fuel` column
yet (v5.3's decoupling lands with the materials patch), so **changing a burn
duration also changes that object's combat health**, and the kindling/furniture
duration ratio is pinned at their hp ratio 8:30 = 1:3.75. Every duration number
in this document inherits that.

### 3.7 `ignition_seed` 0.1 → 0.12 and the headroom

The seed's ONE job is to clear the bootstrap floor with margin. The true dynamic
floors (the lower, UNSTABLE fixed point of the logistic, on the still-air arena)
are kindling/furniture **I_crit = 0.018** and wood **I_crit = 0.090** — wood is
binding, for the `a = 1.0` reason in §3.3. 0.12 clears wood by 1.33× and
kindling by 6.6×.

**Erik's violence headroom, quantified:** the seed sits at 12 % of the intensity
scale; the sustained plateau at 23–26 %; the ceiling at 100 %. The natural
transient peak is 0.65, so a wind-fanned or crowded fire has ~1.5× of live
headroom above its own overshoot and ~4× above its plateau.

---

## 4. GATE EVIDENCE

### (a) KINDLING LIVES THE CAMPFIRE ARC — PASS

Still-air reference arena, natural ignition from `ignition_seed`, no forcing.
CSV: `_fire_tuning_artifacts/pf1b_natural_kindling.csv`.

```
peak I 0.654 @ 29 s      time to 90% of peak  19.8 s     (ramps visibly)
plateau I 0.230          PLATEAU T  271.7 game = 836 K   (band 270-330 ✓)
transient peak T ~ 390 game during the first 30 s
death   496.8 s = 8.28 min      hp 8.00 -> 3.114
PART-BURN  61.1 %   (interim band 30-70 % ✓ ; REPORTED, not gated - Erik's)
DEATH CAUSE (_diagnose)      : knee (T-gate limited: hot=0.148,
                               X_local=0.2028 well above X_ext=0.13)
DEATH CAUSE (counterfactual) : T-GATE-GOVERNED @ t=489.9 s
     bracket -0.04209 | F=0.390 hot=0.217 X=0.2020 I=0.026 |W|=0.0923
     bracket if ambient O2  -0.04045   (oxygen would NOT have saved it)
     bracket if fully hot   +0.01100   (heat WOULD have)
```

**Both classifiers agree: death by the temperature gate, not fuel exhaustion,
with 39 % of the crate left.** That is gate (vii)'s blessed-shape oracle, HARD
requirement, met.

### (b) FURNITURE BURNS LONG — PASS

Same arena, `crate30kg` family (`[materials.furniture]`, hp 30). CSV:
`_fire_tuning_artifacts/pf1b_natural_furniture.csv`.

```
peak I 0.650 @ 27 s      t90 19.0 s
plateau I 0.264          PLATEAU T 281.7 game = 856 K
death   1425.0 s = 23.75 min   (target band 15-40 min ✓)
part-burn 53.5 %   hp 30.00 -> 13.945
DEATH CAUSE: knee / T-GATE-GOVERNED (both classifiers)
```

**THE FLICKER IS BACK, and it is emergent.** The trace oscillates I 0.15–0.48 /
T 210–331 for the whole burn — the honest lag between the fire's own O₂-ring
drawdown, the diffusive refill, and the logistic. Nothing prescribes it.

### (c) LOAD-TIME CHECKS — ALL GREEN

`materials.py::_check_ignition_seed` prints nothing on load. Per flammable
thermal solid (its own formula: claim_faces = 4, AMBIENT o2f), plus the design's
knee check with **β = 0.6 FIXED** and the load-check `I_eq = c·(a − r(1−a))`:

| material | cool_shift | fire_T_ext | gain | I_sustain | need (1.15×) | margin | seed check | I_crit/I_eq | knee ≤ 0.6 |
|---|---|---|---|---|---|---|---|---|---|
| wood | 13 | 100 | 91022.2 | 0.00118 | 0.00136 | 88.1× | **GREEN** | 0.00118 | **OK** |
| furniture | 13 | 80 | 91022.2 | 0.00096 | 0.00111 | 108.2× | **GREEN** | 0.00096 | **OK** |
| kindling | 13 | 80 | 91022.2 | 0.00096 | 0.00111 | 108.2× | **GREEN** | 0.00096 | **OK** |

The TRUE dynamic knee ratios (measured on the arena, not the load-check's
convention) are kindling/furniture **I_crit/I_eq = 0.018/0.26 = 0.069** and wood
**0.090/0.34 = 0.26**. Both far under 0.6; the fire is easy to light and hard to
cool out. Whether that is the feel Erik wants is a session question — the dial
that moves it is `ignition_to_ext_delta`.

### (d) RADIATIVE IGNITION CHAINS — PASS, with the reading stated

One air gap (face-touching is conduction's job and is radiatively INERT under
v7 rule 3). `tools/fire_spread_bench.py`; CSVs in `_fire_tuning_artifacts/`.

| scenario | kindling | furniture | band |
|---|---|---|---|
| **NATURAL end-to-end** (cold source ignited, neighbour watched) | **30.9 s** | **30.0 s** | kindling 30–120 ✓, furniture 30–60 ✓ |
| established burner pinned at the FLICKER PEAK (I 0.45, T 320) | 44.0 s | 44.0 s | ✓ both |
| established burner pinned at the MEDIAN plateau (I 0.25, T 282) | stalls at 268 | stalls at 268 | — |

**The reading.** The one-gap radiative chain is driven by the burner's HOT
EXCURSIONS, not by its median: a receiver pinned against a 282-game source tops
out at 268 and never lights, but a real burner spends part of every flicker
cycle above 320 and the receiver integrates that. The natural end-to-end number
— **30 s, both families** — is the one the game actually produces, and it lands
exactly on Erik's spread ruling ("30–60 s initially"; the earlier bench's 5–13 s
was "way too fast"). `heat_atten` was NOT touched; the levers used were
`T_emit_gate` and `cool_shift`, both sanctioned.

### (e) THE FIRST TRUE SMOTHER CURVE — PASS

Sealed SHIP rooms (`boundary="space"`, no sky refill, no vent), natural burns,
`DRAW_R = 2`. `tools/fire_smother_curve_sweep.py`;
`_fire_tuning_artifacts/smother_curve_summary.txt` + 8 CSVs.

| room | material | death | cause (_diagnose) | cause (counterfactual) | part-burn | room X at death |
|---|---|---|---|---|---|---|
| 6×6 | kindling | **124.5 s** | O2 (X_local 0.1108) | **O2-governed** | 12.5 % | **0.1295** |
| 6×6 | furniture | 123.7 s | O2 (X_local 0.1110) | O2-governed | 3.4 % | 0.1297 |
| 8×8 | kindling | 118.9 s | O2 (X_local 0.1190) | O2-governed | 12.4 % | 0.1533 |
| 8×8 | furniture | 119.5 s | O2 (X_local 0.1182) | O2-governed | 3.4 % | 0.1528 |
| 12×12 | kindling | 114.3 s | O2 (X_local 0.1219) | O2-governed | 12.1 % | 0.1821 |
| 12×12 | furniture | 114.3 s | O2 (X_local 0.1219) | O2-governed | 3.3 % | 0.1819 |
| 20×20 | kindling | 110.3 s | O2 (X_local 0.1217) | O2-governed | 11.9 % | 0.2001 |
| 20×20 | furniture | 111.5 s | O2 (X_local 0.1209) | O2-governed | 3.3 % | 0.1999 |

**BOTH classifiers say O2 in every sealed room** — including the coarse
`X_local <= X_ext + 0.01` one, because §3.1 moved the logistic wall below X_ext
and the flame ring really does reach 0.111.

**The honest boundary, reported rather than smoothed:** room size does NOT flip
the cause. A sealed ship room has no refill at all, so the flame ring
suffocates locally long before the bulk does; what the ladder moves is how much
of the ROOM had to be spent (mean X at death 0.1295 → 0.2001 as the room grows
— that IS the curve). **The control that dies the other way is the planetside
arena with sky refill**: the same crates burn 8.3 / 23.8 min there and die
T-gate-governed. 110 s O2-governed sealed against 500–1400 s T-gate-governed
refilled — that pair is the smother, and it is the ships requirement working.

P-F4b's finding is thereby retired: it reported that every sealed run died in
<1 s at peak I ≈ 0.09 with the room O₂ untouched, and named the
ignition-seed/tempo re-tune as the prerequisite. This is that re-tune.

### (f) THE P-F1a STRICT XFAIL FLIPPED — PASS

`tests/test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood`
XPASSed on the first P-F1b bench run — the designed handoff signal — and is now
a plain passing assertion with the new band. Measured: **the air-separated wood
target ignites at tick 107 (4.5 s) at T = 301.0 game** against wood's 300, with
the burner pinned at flame temperature (the fastest case the chain ever sees).
Its sibling `test_full_chain_radiation_heats_air_separated_wood` reports 300.8
game where it used to report 183.7. All 14 tests in the file pass.

Both chain scenarios were re-anchored to the SHIPPED material table: P-R4's
`cool_shift = 9` pin is deleted, because P-F1b is the tune that pin was waiting
for.

### (g) SUITE + GOLDENS — ONE NEW RED, ROOT-CAUSED, NOT RE-BASELINED

**Without a CUDA build** (apples-to-apples with the recorded baseline):

```
baseline (this branch, pre-patch) : 27 failed, 1857 passed, 27 skipped
after P-F1b                       : 28 failed, 1857 passed, 27 skipped
NEW RED (1): tests/test_w6_armory.py::test_canonical_scenario_golden_and_untouched_rng
RESTORED   : none of the 27 (see below for what WAS restored)
```

**With a CUDA build** the same single root cause surfaces 11 more times: the 11
`tests/test_cuda_*` bit-identity checks that could only SKIP at baseline now run
and each re-asserts the SAME canonical A/B golden. **Their CPU↔GPU parts all
pass** — e.g. `cuda_s3_check` prints "all 45 configs bit-identical" and "CPU vs
GPU water backend: bit-identical over 30 ticks" and then fails only on
`GOLDEN MISMATCH`. 12 names, one cause.

**THE ROOT CAUSE, BISECTED.** Reverting the P-F1b dials one at a time against
the canonical trajectory digest:

```
all 13 dials reverted -> 28678e9d6210533f...  == THE GOLDEN
revert wall_damage    -> 28678e9d6210533f...  == THE GOLDEN
revert any OTHER dial -> 9dbb9cd24bb1551d...  (unchanged from the full P-F1b set)
```

**`wall_damage` alone moves it, and nothing else does.** The task order's premise
("the golden scenario has no flammables") does not hold for THIS golden:
`tests/field_ab_harness.default_scenario_sim` seeds fire on a flammable tile, so
the one dial that governs how fast fuel is consumed necessarily moves the
trajectory. `wall_damage` cannot stay at 0.4 — at 0.4 a 30 kg crate burns for
2 minutes, not 24 — and the only alternative levers (raise `hp`, or decouple
fuel from hp) are respectively a combat-health change and explicitly out of
scope for this patch.

**NOT RE-BASELINED, DELIBERATELY.** The design reserves the arc **one**
deliberate golden rebase, "taken at close, after the dials settle" (v5 §5,
L3-10a/m22), and the dials do not settle until Erik's P-R5 session. So the red
stands, named, with its bisect, as the arc's pending rebase item. It is a
behavioral consequence of an approved behavioral change, not a defect.

**The two goldens P-F1a verified are UNMOVED** — actual digests recomputed by
the same method, since their pass/fail proves nothing (both inherited red):

```
door-present wire-free: 5d944aa8b085fa24a100575a1292196058f15953e0c0726f95342650cb685d8b  (unchanged)
B6 logic loop         : 812c5f80bf66f5caaf546fd8e371f95d504eb993aacc897c360a2701505c99a5  (unchanged)
```

**WHAT WAS RESTORED**: P-F1a's ONE named expected red — the strict xfail (§(f)).
It was not among the 27, so it does not appear in the diff; it flipped from
`xfail(strict)` to a plain passing assertion, which is the whole handoff.

**RE-ANCHORED TESTS** (each documented in place; all now green):

| test | why it moved |
|---|---|
| `test_eos_p4_combustion` ×3 (`e2e_2`, `e2e_4`, `payoff_orderings_perturbation_robust`) | tick budget written for fires that died in tens of ticks; the arms now live for minutes. `max_ticks` 400 → 25000; measured trio **flooded 3090 < vented 3334 < sealed 20008**, ordering intact and still perturbation-stable. The P-R4 `cool_shift = 9` pin deleted (shipped is 13). |
| `test_s3b_fire_determinism::test_fire_field_and_burnthrough_list_bit_identical_run_twice` | the trajectory's last discrete event is the extinguish flip; a held-and-fanned blaze now burns for ~2 min. `TICKS` 90 → 3400 (measured flip at 3205). The determinism claim itself was never in question and passes unchanged. |
| `test_cool_shift_axis` ×2 | asserted every row still carries the seeded global 5. The test's own message said "If this is an intended re-tune … this test must be updated together with a HUMAN-TEST play session" — this IS that re-tune. The cellulosic family is excluded BY NAME; the uniformity assertion becomes the keyed-by-material assertion it was really about. |
| `test_pr3_capacity_law::test_fire_T_ext_is_derived_from_ignition_temp` | pinned Δ = 100 and the two numbers it implies. Now asserts the RELATION and reads Δ off the config. |

### (h) CPU↔CUDA TOLERANCE ZERO ON A LIVE NATURAL BURN — PASS

Both scenes (kindling and furniture), the recalibrated arena, the REAL per-tick
path, 240 ticks each, per-tick SHA-256 over every synced plane the fire arc
touches (`fire`, `temperature`, `wall_hp`, `heat`, `gas`, `wind_x`, `wind_y`,
`atmosphere`, `rad_net`, `rad_amb`, `rad_flux`, `dem_acc`) plus the raw
intensity/temperature/hp at the crate:

```
31 digest lines, CPU build vs cpp/build_cuda build:  IDENTICAL, tol 0
device: NVIDIA RTX 1000 Ada Generation Laptop GPU | sm_89 | CUDA 12.9
```

---

## 5. DEVIATIONS AND OWED ITEMS

1. **`T_emit_gate` = 310, above the suggested 250–280 range.** Forced by gate
   (f): the wood chain must reach 300, and any gate at or under 300 stalls it
   below that. Accuracy cost stated at the key and in §3.5.
2. **The canonical A/B golden moved** (§(g)). One dial, `wall_damage`, bisected.
   NOT re-baselined — that is the arc's single deliberate rebase, owed at close.
3. **The ruling's "12.6 kW ⇒ 890 K" arithmetic assumed χ_bed = 1** (§3.2). The
   honest operating point is ~19 kW; the plateau lands at 836–856 K, inside the
   ruling's own 270–330 game band but below its 890 K headline.
4. **The strict xfail lived in `tests/test_fire_heat_source.py`**, not
   `tests/test_pf1a_radiation_books.py` as the task order said. Flipped where it
   actually is.
5. **Part-burn is REPORTED, not gated** — 61.1 % / 53.5 %. Erik's at the session;
   `ignition_to_ext_delta` is the one dial that moves it.
6. **Fuel is still `hp`** (§3.6). Durations and combat health are one number
   until the materials patch.
7. **`cool_shift = 13` is an interim** (§3.4). F3's convective term re-runs F8
   step 3 and should take most of it back.
8. **Wood is a weak fire and always will be under package A.** `a = 1.0` doubles
   its sky loss, so it plateaus ~240 game against a crate's ~310, and its
   bootstrap floor (0.090) is 5× the crates'. It sustains, it does not
   flashover. `heat_atten` — the only lever — is feel and was not touched.

---

## 6. BENCH CHANGES (calibration instruments, no sim code)

* `tools/fire_spread_bench.py` — **NEW.** The one-air-gap radiative spread
  bench: established-burner (pinned) and natural end-to-end modes, CSV output.
* `tools/fire_timing_harness.py` — `build_level`/`run_one` gain a `material`
  parameter (default `FURN`, so every existing caller is byte-identical) so the
  chartered still-air arena can host the campfire reference object; the seed
  temperature reads the tile's own `ignition_temp` instead of a hardcoded 280
  (same value for both shipped flammable crates); the record gains `w` = the
  plume's own `|wind|` at the fire's tile.
* `tools/fire_room_bench.py` — records `w`; gains `_death_counterfactual`, an
  EXACT death-cause decomposition against the logistic's own growth bracket at
  the TERMINAL crossing (the last tick the bracket was still non-negative):
  would ambient O₂ have saved it, would full heat, or neither. The older
  `_diagnose` is kept verbatim for continuity with the P-F4b package.
* `tools/fire_smother_curve_sweep.py` — room ladder 6/8/12/20, horizon 2400 s,
  reports both classifiers, and its stale "cannot be measured at today's tune"
  finding is replaced by the measured curve.

---

## 7. WHAT THIS PATCH DID NOT DO

No law changed. No new plane, no new term, no new mechanism. `rad_scale` stayed
derived, `burn_rate` stayed anchored, `o2_potency` stayed 1.0, `DRAW_R` stayed
2, every `heat_atten` and `thermal_mass` stayed put, and the wind coefficients
were left for P-F3 where their anchors live. Thirteen numbers moved, each with
one job and a derivation, and the fires came back.
