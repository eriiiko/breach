# DESIGN — fire realism, whole-model critique, multi-material principles (2026-08-01)

**Status: v4 — Q-0 RESOLVED by Erik (2026-08-02): OPTION 2b. See the v4 log
entry at the bottom for the full ruling batch from his annotated read-through
of the plain edition + the two chat discussions. Round-3 (final short review)
running on this version. Previous status follows.**

**Status: v3 — two adversarial rounds synthesized (8 lens-reports, ~180 findings;
verdicts + indices: `fire_realism_critiques_round1_2026-08-01.md`,
`fire_realism_critiques_round2_2026-08-01.md`). The keystone algebra is now
CONVERGED (two lenses derived the identical correction independently — theorem
T1). Round 2 also discovered the design's true open keystone: THE AIR-SUPPLY
CHANNEL (theorem T2). v3 is therefore decision-ready in two stages: §3's
mechanics are buildable as specified; §0's Q-0 CANNOT be ruled until the supply
pre-measurement runs. NOT BLESSED; nothing builds.**

Erik's commission and the standing iron rules as in v1/v2 (Q16.16, no libm,
CPU↔CUDA tol 0, HUMAN-TEST, one-job dials, named lumps with references).
Package naming: "package B" in older indices ≡ REAL.

---

## 0. The energy books, the supply theorem, and Q-0 reframed

**Units (canonical, verified twice independently):** 1 heat count ≡ 1.968e-4 J
(≡ rad_scale bake: σ·A_face·dt with A = 0.833 m², dt = 1/24 s — which is why
`rad_scale` is DERIVED, never authored: `rad_scale ≡ σ·A·dt/(J/count)`, and the
J/count anchor is pinned once, from the chemistry side). Furniture effective
C = 51.6 J/K (the deliberate ~140× δ-layer lump). Operating point cited from
`tune_r5_lone_wd020.csv` (post-P-R4): peak I 0.192 / steady_T 385 / peak T 544;
the books below quote the range across the artifact set, not one flattering tick.

| channel at the blessed burn | power |
|---|---|
| chemical (burn_rate 0.02) | 4.85–6.7 kW |
| H_bed to the bed | 2.6–3.6 kW (χ_bed ≈ 0.54 ✓ of Huggett) |
| honest sky sink at ε = 0.5, T 385–544 game | 30–45 kW |
| + h-anchored convection (F3) at the plateau | 15–29 kW |
| **required HRR for the blessed plateau, χ_bed ≤ 1** | **60–97 kW** |
| real crate fire (Babrauskas) | 100–250 kW |

**T2 — the supply theorem (round 2, measured from our own bench):** diffusive
O₂ transport to one burning tile ≈ 0.105 units·s⁻¹ per unit of X-depression,
and the depression is capped at 0.08 (X_ext) ⇒ **maximum sustainable delivery
≈ 27 kW (40 kW only at the point of death)**. The engine is 2D and gravity-free:
there is NO buoyant entrainment, and the fire's own plume pushes air OUTWARD.
Real 100+ kW crate fires breathe by entrainment. **Therefore neither v2 package
closes: small books cannot spread fire by ANY channel (radiative flux 24× under
the 12 kW/m² ignition threshold — physical, not artifact; crate↔crate
conduction is exactly zero in-engine, furniture κ = 0), and big books cannot
breathe (achievable HRR ~27 kW, with I halving toward the knee).** The
controlling channel of the whole design is the air supply, absent from every
prior version.

**Q-0 (reframed): where does the fire's air come from? [ERIK — after the
pre-measurements]** Options, with my recommendation:

- **R1 — RECOMMENDED: REAL books + a designed supply channel.** Re-anchor
  `burn_rate` per fuel class (the "literature act" §4 sanctions — NOTE this
  COMPLETES Erik's D1 ruling rather than reversing it: the ×10 he declined was
  an uncited hack; ×25-with-Babrauskas is cited; and REAL RETIRES the D1
  accumulator, demand ≈ 23 counts/tick, while small books must keep it) — plus
  ONE new supply term standing in for entrainment (candidates, own mini-design:
  a burning-tile inflow bonus on the O₂ advection; or a vent/A√H-law room
  supply; the term is the missing physics and gets its own design-gate).
  Payoff: honest 5–30 s radiative spread (44.5 kW ⇒ ≈13 kW/m² at one face —
  REAL *reproduces* the ignition anchor exactly), real smother mechanics, real
  room heat, plateau lands 1140–1310 K in the flame band with no new dial, fuel
  energy ≈ 3 kg wood (real crate 10–30 kg; small books imply 120 g).
- **R2 — small books + gas-mediated spread:** keep 5 kW-scale fires; spread
  moves to F3's hot-gas contact (the physically correct short-range mechanism
  at this scale) and radiative ignition is declared out. Cheapest; changes the
  feel of firefighting and clusters materially.
- **R3 — the soot-gas counterparty** (round-1 L3's fork): the smoke plane
  carries emissivity; the hot layer absorbs and re-radiates. Physically the
  dominant real-compartment channel; the largest build; composes with R1.
- v2's "package GAME" is retired as arithmetically infeasible (round-2 L3-1,
  L1 R2-7): its two sizing conditions were one degree of freedom, and its
  fallback spread channels measure zero.

**Pre-measurements that gate the ruling (cheap, no law change, run first):**
(i) the SUPPLY BENCH — max sustainable O₂ delivery to one burning tile at
X_ring ≥ 0.17, still air, across F4 room sizes and vent states; publishes the
supply constant k and the |W|↔m/s conversion; (ii) the χ_bed CONSISTENCY CHECK
`H_bed·(J/count)/4.83 MJ ∈ [0.3, 1.0]` at every calibration point (pure
arithmetic — already verified 0.54 at the incumbent); (iii) the sub-question
Erik must feel, stated plainly: **is seconds-scale radiative spread the feel
you want?** (5.0/11.2 s were bench gate results, never human-tested.)

## 1. Problem inventory — unchanged from v2, plus the round-2 discoveries

P1–P9 as v2, with: P3 (cluster mutual feeding) MOVED to accepted-gaps until an
intensity-feedback channel exists — under the current law `hot` saturates and
clustered tiles compete for O₂, so net feeding is negative (R2-10); the honest
F1 payoff for clusters is longevity + ignition, not intensity. NEW P10: the
air-supply channel (T2). NEW P11 (shipped bugs found by round 2, fix
independently of any ruling): the `dem_acc` stale-debt skip paths
(`combustion.cpp:168-173` — reset the four slots; synced digest state) and the
`H_bed` int64→int32 deposit clamp firing on full-drain cells (re-derive
`n_floor_heat` against THAT path, not the gas divisor). Also reclassified: the
per-source `mt19937` is a dormant cross-machine desync landmine (delete the
distribution, not just the construction).

## 2. Whole-model critique — v2's lists plus the round-2 channel

Add to *wrong or missing*: **air supply/entrainment (T2 — now the design's
open keystone)**; the radiative-feedback-to-burning-rate channel (ṁ″ ∝
q̇″_inc/L_v) stays the named carrier of true mutual feeding and flashover.
Add to *deliberately lumped (named)*: the gas heat capacity lump (model gas
tile 6.45 J/K vs physical 242 J/K — **37×**; consequence: room warming ~37×
too fast unless gated on the TIME CONSTANT, not just the steady value — R2-17);
the touching-solids interface (conduction is the physical channel; the
radiative pair is SUPPRESSED where `face_shift ≠ NO_FACE` — R2-6b); sub-gate
emitters (below `T_emit_gate` a tile pays sky loss regardless of enclosure —
error bounded by E°(gate) − E°(0) ≈ 1.7 game/tick; the INTERIOR-WALL mask
kills the buried-tile absurdity: tiles whose four neighbours are all solid
skip the sink).

## 3. The fixes (round-2-corrected; all formulas final unless marked [F8-n])

### F1+F2 — THE HONEST RADIATOR (converged books — theorem T1)

**One emission entity per tile** (R2-4): `E_emit(i) = φ·E°[min(T_cap, T_i +
flame_lift·I_i)]` — burning tiles carry φ/flame_lift per Q-0's books (under R1:
φ ≡ 1, flame_lift ≡ 0 — F1 returns to ZERO new dials); non-burning solids emit
at bed T. Sink, pair and credit ALL read this entity.

**(a) The ambient sink** — per thermal solid, per tick, in Pass 1 (NOT via the
const `rad_net` plane — L2-B2): int64 `sink = sink_coef_q[i]·(E_emit(i) −
E°[0])` with `sink_coef_q = quantize(a_i·Ω)` baked per material (M10, keeps the
TU integer), Ω ≡ Σw = 1 (ONE face-equivalent — the word "sphere" is banned;
L4-8), clamped by the SINK'S OWN limiter (R2-5: bind point must sit above the
operating band; print first-bind T per material at the gate), converted
`shr_round0(·, his)`, applied with sat_add + HIGH rail + **LOW rail at 0 with a
counter** (carried by the budget argument: sink + 8 pairs ≤ 9T/16 < T). Pass-1
order PINNED: sink → rad fold → heat deposit, rails at each (m14). Skipped on
interior-wall tiles (mask above). `rad_amb_flux` ledger: per-tile plane, plain
wrapping adds, host-side uint64 reduction (m13/m20 — no global atomic).

**(b) Pair + credit (T1):** per absorbing cell r on a ray from s:
`rad_net[r] += pair` with `pair = a_s·a_r·τ·w·(E_emit(s) − E°[T_r])`;
`rad_net[s] += credit` with `credit = a_s·a_r·τ·w·(E°[T_r] − E°[0])` — the
credit REPLACES the emitter's pair debit (v2's double-charge is the round-2
blocker; the corrected books satisfy: ambient scenery ⇒ exactly the open-air
sink [Erik's equivalence]; equal-T full view ⇒ zero net; partial view ⇒ loss
through uncovered directions only). **Mandatory exclusion: no credit at the
source's own cell** (distance 0 — else a_s² of the sink self-refunds; R2-2).
Pair SUPPRESSED across conducting interfaces (`face_shift ≠ NO_FACE`).
Contract note at the site: per-pair antisymmetry is retired; conservation is
the ledger identity, with the two sides accumulated at separate sites (M7).
Credit rides the emitter's cast ⇒ sub-gate tiles get no refund: `T_emit_gate`
is now an ACCURACY dial (hysteresis pair 190/170 kept; the boundary gate below
tests both sides of it).

**(c) Opaque termination (the ε-retune's law half — R2-15c/M9):** a ray
terminating on an opaque solid credits the source the FULL remaining direction
share `a_s·τ·w·(E°[T_r] − E°[0])`-equivalent (cavity closure — else low-ε
sealed rooms shred energy). This changes the heat-touched tile set: it is a LAW
BULLET with its own gate and CUDA twin, not an F8 parenthetical.

**(d) Tables and rails:** `E°` widens to **int64** (its int32 saturation at
~1768 game — 9× below T_MAX_PHYS — was the real ceiling, uncounted; L2-B3);
the fold's aggregate is clamped ONCE at conversion to a stated fraction of T_i
with a counter (M6); `raycaster.h`'s overflow bound re-derived for
sink+pair+credit; `range_base`/`range_per_intensity` are now ENERGY-BOOKS
parameters — owned in F8 step 2 with the implied sky-fraction stated, and the
range∝I feedback named (L2-B5). φ/flame_lift enter the bake cache key or ride
per-source multiplies (m16).

**(e) Gates:** (i) equivalence, pinned single-tick, walls at T = 0, tolerance
= (n+1)/2 counts + the geometric defect (or an a = 1.0 geometry where
telescoping is exact) (M8); (ii) ledger `Σ rad_net + Σ rad_amb == 0` with its
detection scope stated (int32 wrap; clamp/booking mismatch — it CANNOT catch
wrong credits: that is gate (iii)'s job) (M7); (iii) uniform-equal-T grid ⇒
`rad_pair_dbg ≡ 0`, via a nullable debug plane (rad_flux idiom, null in
production); (iv) the `T_emit_gate` BOUNDARY test: a warm tile in a sealed
equal-T box shows |net| ≤ ε just below AND just above the gate (R2-3); (v) the
SHRINKING-ROOM test: `rad_amb_flux` falls measurably as the F4 room closes in
(B5 — the cavity actually works); (vi) knee re-measure: I_crit/I_eq lands in
[0.3, 0.7] or the knee is re-sited with Erik (levers: `ignition_to_ext_delta`,
`fire_T_span`; the cool_shift LEVER-RANGE amendment is hereby retired — its
dial is spent by the F8 ownership split); (vii) the BLESSED-SHAPE oracle:
lone crate still-air dies BY THE KNEE, not the fuel floor (HARD); the
part-burn fraction is ERIK'S at the session (incumbent measured: 61.7%);
(viii) cluster scenarios, DEFINED in F4 (corner-seeded 2×2 still-air = the
big-log fizzle, must fizzle; igniter-adjacent stack = spread case [Q-0]);
(ix) CPU↔CUDA tol 0 step+resident. P-F1a acceptance, stated in the P-R2 form:
"at frozen dials every bench fire extinguishes (plateau ≈ 142 game < ext 180);
the named fire-suite reds are expected and listed; P-F1b restores them —
accepted." 

### F3 — CONVECTIVE EXCHANGE (one added factor; owns room heating)

As v2 (per-face signed inflow — one axis component, negate + max, NO sqrt
[m15]; global CONV_SHIFT h-anchored; signed `conv_net` plane; own convex pass
with both-ends budget, gas end floored, counter on the n_floor engagement
[m17]) **plus the round-2 factor: `q_f *= min(FP_ONE, N_j)`** — density-scaled
convection (R2-16). This one multiply: bounds the thin-gas amplification (the
decisions-#16 runaway stays closed), makes convection vanish into vacuum by
construction (a breached room does not cool by "convection to nothing"), and
**retires `cool_shift_vacuum`** (its radiative justification moved to F1; its
convective one now emerges — F8 step 3 removes the offset with a written
rationale). Room-heat acceptance: ΔT_gas bands at named matrix points AND the
time constant (the 37× gas lump is stated at the site; `c_v ≡ 1` stays a
lumped convention with the lump written down) (R2-17). `w_gain`: owned by F8
step 7, one stated anchor, or deleted in favor of the clamp alone.

### F-BO — BLOW-OUT (exact integer form; falsifiable scaling)

`inv_c_eff = mul_q16(INV_C, FP_ONE + mul_q16(k_strain_q, W_in))`, clamped at
`STRAIN_MAX_Q` (shared with F3's clamp so the two consumers of the inflow
measure agree at blast spikes) — the EXACT c/(1+kW) law, no divide, no
linearised fallback; byte-identical at k_strain = 0 (free regression oracle)
(L2-B4). Anchor 1 (marginal fire dies at W ≈ X) SOLVES k_strain; anchor 2
(established fire survives Y) is a FALSIFICATION TEST with the prediction
written: Y/X = (r_est − 1)/(r_marg − 1), r = I_eq/I_crit — if the bench
refuses it, the named cause is the missing fire-size scaling (Heskestad
L ∝ Q^0.4), a design return, not a re-tune (R2-12). Prerequisite: the
|W|↔m/s conversion published from an F4 vent run. THE THIRD ANCHOR (L3-6,
the ruling's own acceptance): "a wind-fed lone crate CROSSES the knee and
burns out at some plausible W" gets its own P-F3 gate, with the ruling's
escape clause verbatim: if the supply channel alone proves too weak, re-siting
the wind fan is a DESIGN question — bring it back, do not dial it.

### F4 — SCENARIO BENCH (first patch; now also the pre-measurement instrument)

As v2, plus: the SUPPLY BENCH mode (T2's pre-measurement — room-size × vent
sweep publishing the supply constant and the |W|↔m/s map); defined cluster
layouts (gate viii); the O₂-axis variance matrix (volume × leak/permeability ×
fuel load — BOTH smother and fuel-exhaustion outcomes required; L3-9); the
ML/zombie fire-dominance scenario promoted to an OWNED acceptance in F8 step 8
with a criterion Erik can rule on (L3-8/M9). Air-cell honesty note where
smother timescales are quoted: the cell inventory implies a ~0.34 m ceiling;
a 3 m room is ~9× slower (R2-11).

### F5′ — hot-layer unit damage: as v2, with the max taken on the INTEGER side
before conversion (m23). Erik's Q-F.

### F6 — smolder: DEFERRED to its chartered mini-design (unchanged from v2).

### F7 — payloads: unchanged from v2 (accelerant-deposit named as the future
form; re-test post-keystone).

## F8 — CALIBRATION (v3: the two hidden arrows removed)

0. **Q-0 ruling** (post pre-measurements) fixes the watt books AND outputs
   `T_flame_ref`. 1. **ε-retune** — now explicitly a LAW PATCH (march change:
   opacity decoupled from ε via F1(c); steel 0.2–0.4, glass ≈ 0.9 thermal-IR,
   furniture/wood → 0.9; occlusion-feel deltas HUMAN-TESTED). 2. **rad_scale is
   DERIVED** (formula in §0; authors nothing — R2-13); its companions
   `range_base`/`range_per_intensity` get their sky-fraction statement and the
   shrinking-room gate. The 12 kW/m² gap-1 flux becomes a step-4 PREDICTION.
   3. cool_shift + CONV_SHIFT joint (pinned-T_gas protocol; predicted outcome
   stated in advance: furniture's residue ≈ 0 ⇒ retire, not re-fit;
   cool_shift_vacuum retired with rationale). 4. H_bed re-solve + the χ_bed
   check [0.3, 1.0] + the CONSISTENCY TEST |T_plateau − T_flame_ref| < band
   with a declared two-pass iterate (R2-14) + `n_floor_heat` re-derived against
   the H_bed clamp path (m21). **4.5 THE SUPPLY CHECK (T2): if the plateau
   solve demands more kW than the measured supply delivers, the package is
   falsified and the arc STOPS for the supply-channel design — the cheapest
   gate in the document, runnable before any patch.** 5+6. knee + duration,
   JOINT under R1 (the O₂-coupled drain reaches ~40% of hp flow; expected
   `wall_damage` direction stated per package) — Erik's session. 7. k_strain +
   w_gain anchors (AT P-F3 TIME, not P-R5 — L3-10c). 8. room curve (value +
   time constant + both matrices) + the ML/zombie acceptance + substep-cap
   ASSERTIONS (n_smoke, N_SUB — a hit cap is a silent CFL violation, not a
   cost note; m22).

## 4. Multi-material principles — v2 hardened (round-2 edits only)

The unit identity replaces the 2× warning: **`ignition_temp_game ≈
°C_literature − 10`** (L4-18). Validity bound in shipped columns: hp within 2×
of 60, thermal_mass within one step of 8 (L4-19). The knee LOAD CHECK: stated
as a monotone table search over the 3-term balance (sink + conv + cool vs
deposit), β = 0.6 FIXED, flammables only, **WARN at P-M1, promoted to ERROR at
P-R5 close** (M11/L4-4). Cellulosic band ≡ the lone-crate band Erik sets.
Rails enumerated (T_MAX_PHYS, sink limiter, aggregate fold clamp,
n_floor_heat, sat_add). Everything else as v2 (sorting rule; cellulosic+none
ship; reserved classes are load errors; fuel_per_o2 in class 3; anchors stay
anchored; L_v + geometry as named future lumps).

## 5. Execution (resized; NEW-ARC declaration)

**This design is a NEW ARC: the P-R arc's one golden rebase was spent at
P-R4/D2. This arc carries its own single deliberate rebase, taken at close,
after the dials settle** (L3-10a, m22). Every new plane (conv_net,
rad_pair_dbg, the ledger plane) is per-tick-wiped and digest-excluded —
persistent synced state forfeits the freeze (M12); an ARC-LOCAL golden is
re-baselined per patch with one-line rationales so bisection survives the
frozen-canonical window. Order: **P-F4** (tools + pre-measurements) →
**Q-0 ruling (Erik)** → **P-F1a/b** (law at frozen dials, then F8 1–4 with
step-1's march change gated) → **P-F3+F-BO** (anchors solved here) → **P-F5′**
(if commissioned) → **P-M1** (classes; knee check WARN) → **P-R5** (Erik:
knee/duration/room/feel + the fraction numbers + knee check → ERROR) → canon
fold + archive. Erik-touchpoint table: Q-0 ruling · wind anchors (P-F3) ·
ε/occlusion feel (P-F1b) · Q-B..Q-F rulings · the P-R5 session · ~5 play
tests. CUDA twins named per patch as v2.

## 6. Open questions for Erik

Q-0 (reframed, §0 — after the pre-measurements; includes the spread-feel
sub-question) · Q-B smolder mini-design commission (rec: yes, post-P-F3) ·
Q-C doors both-rows (rec: with P-M1) · Q-D room-feel targets = value + time
constant bands at matrix points (rec) · Q-E class config shape (rec:
`[fuel_class.*]` sections) · Q-F F5′ now vs gap (rec: with P-F3) ·
**Q-G (new): the supply-channel design fork under R1** (inflow bonus vs
vent-law vs full entrainment mini-design) — Erik picks the shape, the
mini-design fills it.

## 7. Accepted gaps — v2's list plus:

P3 cluster INTENSITY feeding (until the ṁ″ ∝ flux/L_v channel); gas radiative
participation (grows under R1 — soot/gas radiation is 20–40% of real
compartment transfer; belongs in R3's scope); sub-gate emitters radiate to sky
regardless of enclosure (bounded, stated); reflective-metal specularity
(handled to first order by F1(c) cavity closure; angular effects stay lumped);
the 37× gas heat-capacity lump (stated, time-constant-gated); air-cell 0.34 m
ceiling (stated where sold).

## 8. Citations — as v2, plus Heskestad (flame height, F-BO's falsification
scaling) and Babrauskas under R1.

## 9. The dial ledger (honest form — L4-1)

| | entries |
|---|---|
| **Born (authored)** | CONV_SHIFT (h-anchored global), k_strain (anchor-solved), w_gain (one anchor or deleted), STRAIN_MAX_Q + WIND_MULT (stability constants w/ derivations), [R1: class burn_rate factor + the supply term's constant(s) — priced in Q-G], [F5′: 2] |
| **Born (derived, formulas stated)** | rad_scale, Ω ≡ 1, φ/flame_lift (≡ 1/0 under R1), I_crit[mat], β = 0.6, sink_coef_q, T_flame_ref |
| **Dies** | k_wind_fan, k_wind_strip, cool_shift_vacuum, [R1: the D1 dem_acc plane retires] |
| **Retunes (re-authored values)** | per-material heat_atten (ε), per-material cool_shift, H_bed, ignition_to_ext_delta, fire_T_span, wall_damage, fuel_per_o2 (class base), n_floor_heat, range_base/range_per_intensity (now energy-books), [R1: burn_rate] |
| **Net authored** | **+3..+5, minus 3–4 deaths — approximately flat, with P1/P2/P4/P5/P9/P10 addressed and P3 honestly gapped** |

---

## Critique log

- v1: round-1 panel — 14 blockers; archive round1 file. Structurals: energy
  books (→ §0), residual → sink/credit, F6 deferred, F5 deleted, wind rebuilt.
- v2: round-2 panel — keystone algebra CONVERGED (T1: v2's credit form
  double-charged; corrected books derived identically by two lenses; self-cell
  exclusion mandatory); **T2: the air-supply channel discovered as the true
  open keystone — Q-0 reframed, pre-measurements specified**; ~40 mechanical
  repairs folded (this doc). Archive: round2 file.
- v3 (this): synthesis. NEXT: the F4 supply pre-measurements → Erik's Q-0/Q-G
  session → round-3 spot-check of §3's corrected algebra by one fresh lens →
  blessing → execution per §5.
- **v4, 2026-08-02 — Q-0 RESOLVED + the full ruling batch (Erik's annotated
  read of the plain edition + two chat discussions). BLESSED: the decision
  package.** The rulings, all binding:
  **(1) THE SUPPLY MECHANISM IS OPTION 2b** — Erik's extended oxygen draw:
  burning tiles consume O₂ from open cells within a small radius (2–3,
  distance-weighted), reached THROUGH CONNECTED OPEN CELLS ONLY from each open
  face — never through solids (a wall breathes only via its open faces,
  extended outward; walls hold no pore gas, crates do). Deterministic
  generalization of the existing per-air-cell demand share. This is the
  entrainment stand-in: it raises DELIVERY without inflating room O₂
  inventories, so sealed-room smothering stays exactly real (the ships
  requirement). **NO O₂-potency factor** ("we try without making O2 more
  potent") — revisit only with sealed-room bench evidence, as a bounded
  topping, if 2b measurably falls short. Fire power target: flame-look power
  sized to the MEASURED supply-vs-draw-radius curve (F4 bench gains that
  measurement + a sealed-room smother check at the chosen power). R3 (smoke
  radiation) layers later; Erik's slow-tick compounding-ray idea recorded in
  its folder; his cost fear shared.
  **(2) BURN DURATIONS ARE PER-MATERIAL-FAMILY; the 6–8 min universal band is
  RETIRED.** Erik: a 30 kg crate burning ~30 min is desirable. The TUNING
  REFERENCE becomes a campfire-scale fuel object (1–3 kg class — the scale his
  feel-verdicts were always calibrated against). Future furniture family
  (crate10kg/crate30kg/table…) spans kindling-quick to siege-slow.
  **(3) SPREAD IS CONDUCTION-LED** (crate conductivity: YES — the primary
  spread mechanism, 30–60 s initially, faster for established fires; the
  earlier bench's 5–13 s was "way too fast"); **RADIATION IS THE FLASHOVER
  CHANNEL**, tuned to bite for sustained burns. First ignition from cold
  mimics campfire-starting (minutes-scale via ignition_temp dynamics).
  **(4) AIR IS HEATED AT THE FIRE ONLY** — own tile (pore gas) + open faces;
  NO vicinity radius; advection spreads it. The black-body render channel
  reads air temperature, so this also restores the lost hot-air visuals
  (Erik's correction: air temperature WAS the beautiful part of fire's look).
  **(5) T_ambient = room temperature (game 0)**; space-cold directions after
  hull breach = named later refinement.
  **(6) The requirement-1/9 rephrasings** (deterministic part-burn mechanism;
  sensitive-dependence-not-randomness), **crates stay immovable tiles** with
  freely-set conductivity (movable entity-furniture later, own simplified
  fire), **doors become a per-material family** (burnable and not),
  **hot-air unit damage YES** + gear-determines-safe-range logged for the
  unit system, **grenades join the payload re-review**, **one-tuning-night-
  per-material accepted** (extendability demoted to convenience), **ember
  mini-design slotted immediately after the wind fixes** (depends on Fix B;
  Erik's blow-to-rekindle is its centerpiece), **classes config designed when
  wood is perfect**, **Python prototyping allowed as a tool**, the zombie
  claim corrected (fire effective but NOT the only permanent stopper).
  **(7) The wind story** (reality: flame-stripping + fuel-surface cooling;
  engine history: the pre-EOS prescribed strip term and its self-plume
  poisoning; what is emergent today [fanning via O₂ supply] vs designed
  [Fix B cooling = emergent given one anchored coefficient; Fix C strain =
  prescribed-in-shape, physically justified] vs impossible-to-recur
  [inflow-only measure]) was delivered and accepted; the FLICKER Erik loved
  is expected to return via three honest lags and is a named feel item.
  **NEXT: round-3 short review on this v4 → commit both docs → F4 bench +
  supply/radius measurement → execution per §5 (bench first).**
- **2026-08-02, Erik inputs (folded into the PLAIN EDITION —
  `fire_realism_design_plain_2026-08-02.md`, now the decision copy; this file
  remains the engineering spec and gets its v4 pass after his session):**
  (1) SPACE SHIPS: the sky-exchange refill is planetside-only — most levels
  have NO sky; supply design + T2's bench must lead with sealed/vented ship
  rooms; the sky is the special case. (2) THE SCALING RULING (new REQ-11):
  tiles are not massive — effective mass/fuel/ignition-sensitivity are
  choosable within reason; this legitimizes thin-fuel ignition scaling ("the
  cheat" is physics) and opens **Q-0 Option R-SCALED**: size the fire to the
  measured supply (~campfire scale), keep the flame-hot look, scale ignition
  sensitivity + add modest crate conductivity for spread — NEW RECOMMENDED
  default, pending the supply bench + Erik's spread-speed feel verdict
  (seconds vs ~half-minute vs minutes — asked explicitly). (3) CRATE
  CONDUCTION: Erik leans yes (κ=0's bench rationale expires post-F1/F3);
  movable "entity furniture" stays a future separate system. (4) THE
  REQUIREMENTS LIST is now canonical in the plain edition §1 (campfire arc,
  O₂ both ways, wind both ways, SHOCKWAVE-fire interplay, FLASHOVER as the
  named form of Erik's "over-ignition", spread, ships/rooms, materials,
  variance, determinism, scaling). Q-0's packages map: R-SCALED ≈ refined R2;
  R1 (REAL+supply) stays the big-payoff path; R3 unchanged.


---

## v5 — THE CLOSURE BLOCK (2026-08-02, after round 3; AUTHORITATIVE over any
## conflicting text above; implementation agents read THIS first)

Round-3 verdicts: `fire_realism_critiques_round3_2026-08-02.md`. This block
closes every blocker. Language above that presents Q-0 as open, uses the
R1/R2/R3/GAME names, the 6-8 min band, the lone-crate reference, or stale
gate lists is SUPERSEDED by v4's rulings plus this block.

### v5.1 — THE RADIATION BOOKS, FINAL FORM (fourth and structural)

Lesson of three failed attempts: conservation must be STRUCTURAL, not
algebraic. **The invariant: every integer added to or subtracted from any
radiation plane is simultaneously booked to the ambient ledger with the
opposite sign, at the same code site, post-clamp.** `Sum(rad_net) + rad_amb
== 0` then holds by construction through every clamp, mask, refund and rail;
only the four PHYSICAL limits remain gated properties. The terms:

- **Sink** (per thermal solid, Pass 1; interior-wall mask skips fully-enclosed
  tiles): `s_i = clamp(sink_coef_q[i]*(E_emit(i) - E0[0]), sink_budget_i)`;
  temperature path per v3; ledger `+s_i`.
- **Per absorbing marched cell r on a ray from emitter s** (SELF-CELL WHOLLY
  EXCLUDED — pair AND credit; the distance-0 source-cell deposit books
  nothing):
  - r NOT an emitter (mask plane; integer threshold = quantize(T_emit_gate);
    membership = burning OR thermal_solid at/above the gate — the same set
    that casts): `pair = clamp(a_s*a_r*tau*w*(E_emit(s) - E0[T_r]))`;
    `rad_net[r] += pair`, ledger `-pair`. `credit = a_s*a_r*tau*w*(E0[T_r] -
    E0[0])` (emitter-budget clamped); `rad_net[s] += credit`, ledger
    `-credit`.
  - r IS an emitter: **one-way** `pair = clamp(a_s*a_r*tau*w*(E_emit(s) -
    E0[0]))`; `rad_net[r] += pair`, ledger `-pair`; **NO credit at s** — s's
    inflow from r arrives on r's own cast. (Kills the mutual-emitter 2x and
    the gate-crossing rate doubling — round-3 F1.)
- **Refunds** (each: `rad_net[s] += refund`, ledger `-refund`):
  - opaque termination (F1(c)): remaining direction share vs the ambient
    potential, as v3;
  - **suppressed conduction interface** (face_shift != NO_FACE — now live for
    crates under ruling 3): refund s its OWN share `a_s*tau*w*(E_emit(s) -
    E0[0])` and terminate the ray — the interface is radiatively inert;
    conduction owns it (round-3 F11);
  - **range/cull termination**: the same own-share refund (a ray that dies of
    reach must not book its remainder as sky loss inside a sealed room —
    round-3 F3). Consequence: `range_base`/`range_per_intensity` lose their
    energy-books role (reach is geometry only); F8 step 2's sky-fraction
    statement is superseded.
- phi/flame_lift under the 2b package: ALIVE, sized by the watt books at the
  measured supply (F8 steps 0/4); the source-cell exclusion covers the PAIR
  too (else the flame deposits its own lift into itself — round-3 F4); gate
  (iii) runs on a FIRE-FREE uniform grid.
- Gates restated: (iv) becomes TWO cases — (a) sealed equal-T box just below
  and just above the gate (|net| <= eps both sides); (b) sealed
  TWO-TEMPERATURE box, both tiles above the gate: measured exchange equals
  the one-way law at 1x (catches the 2x), continuous across a gate crossing.
  (v) adds a sealed equal-T room WIDER than max_range (exercises the range
  refund). The ledger gate's meaning: any red = a term booked at one site
  and not the other, by construction.

### v5.2 — F-O2b: THE EXTENDED DRAW (full specification)

- **Neighbourhood:** open cells within BFS hop-distance <= R (ship R = 2;
  R = 3 is the sweep's upper point) from the burning tile, expanded through
  open cells only, **path weight permeability-multiplicative** (product of
  min(perm) per hop — the existing physics_engine idiom): a crate (perm 0.5)
  attenuates the draw through itself, a wall (perm 0) blocks — resolves
  round-3 F8; vacuum cells terminate expansion. Enumeration is a canonical
  fixed-offset unrolled relaxation (R rounds) — NO queue, NO truncation;
  MAX_CLAIMANTS per air cell is a hard assert (a hit cap is a violation).
- **Demand:** per (burning tile, reachable cell j): `dem = burn_cap_q * I *
  o2f_j * W_hop[d(j)] * w_path`, with `W_hop` a BAKED integer table (ship
  W_hop = quantize(1/(1+d))). Written falsifier for the bench: quasi-steady
  delivery scales with the draw BOUNDARY (~2-3x at R = 2-3), not the area.
  The per-air-cell demand-share allocation, full-drain rule, per-cell O2
  floor and lowest-index tiebreak generalize UNCHANGED (order-free —
  round-3 verified). D1's dem_acc generalizes per (cell, slot); its reset
  rule carries over.
- **Deposits re-sited (round-3 F5, honoring ruling 4):** combustion heat and
  soot do NOT land at distant donor cells. Drawn O2 is booked at the donors;
  the heat/soot deposit lands at the FIRE'S OWN tile + its open faces via a
  second gather keyed like alloc_face (offset-keyed per-tick plane;
  single-writer preserved; mallocs/memsets priced on both backends). Air
  heats at the fire only; the hot-air visuals follow the fire.
- **The o2f SENSOR stays radius-1** (the fire's own ring) — a DELIBERATE,
  STATED lump: knee, extinction and smother semantics keep their meaning;
  2b raises DELIVERY (fuel throughput, HRR), not the local sensor. The
  supply sweep measures total drawn counts/s at steady intensity with the
  ring-by-ring X profile published (round-3 F7/F10).
- **New authored dials** (§9 re-priced): `DRAW_R` (2) and the `W_hop` form
  (one choice, bench-falsified). CUDA twin: `cuda_combustion.cu` (plane
  layout + register budget per round-3 F6). Digest: the offset plane is
  per-tick-wiped; dem_acc's widened layout is a digest-spec version bump,
  scheduled WITH the patch; the arc-local golden carries it.
- **Execution order (replaces §5's list; round-3 Q4):**
  P-F4a (bench tooling + CAMPFIRE REFERENCE OBJECT: new material row in the
  1-3 kg class + tile id + scenario; + STILL-AIR REFERENCE arena with its
  tuned-parameter list; + baseline diffusion measurement) -> P-O2b (this
  law, frozen dials) -> P-F4b (supply-vs-radius sweep + sealed-room smother
  check + FORCED-WIND level with a literature slot) -> ERIK: radius +
  fire-power sizing call (new touchpoint) -> P-F1a/b (v5.1 books; eps
  retune) -> P-F3+F-BO (wind anchors here) -> P-EMBER (mini-design + build;
  its charter answers the second-threshold question) -> P-F5' -> P-M1
  (classes recipe + door family + airlock-fire test + furniture-to-crate
  rename) -> P-R5 (Erik; class-config shape settled here) -> P-PAYLOAD
  (molotov + grenades) -> canon fold + the single arc rebase (dials settled,
  substep caps asserted).

### v5.3 — Rulings-fold completions (round-3 fidelity lens)

**Fuel decouples from combat hp:** new per-material `fuel` column (the
fire-energy store; hp stays combat health) — RECOMMENDED-ADOPTED pending
Erik veto; enters Fix E box 2 and §9 Born. Effective fuel != nominal mass,
stated plainly: the 30 kg crate carries roughly 50 MJ effective (~3 kg of
actual burn) and may burn ~30 min at the measured supply. **Burn-duration
bands are PER MATERIAL** (derived from the fuel column); "cellulosic band ==
the lone-crate band" is deleted. **The blessed-shape oracle** re-points to
the campfire reference object (hard death-by-knee THERE only; other
families: monotone heavier-implies-longer with the arc shape preserved;
kindling may legitimately die at the fuel floor); it runs AFTER P-O2b; the
61.7% figure is a historical pre-fix label. **Gate (viii) re-derived under
crate conductivity:** a touching corner-seeded 2x2 SPREADS at 30-60 s per
hop and dies by the knee at the cluster edge; the fizzle case moves to the
SPACED 2x2 (the radiation-only geometry). **Erik's §2 blessings recorded:**
the constant heat split is a named lump (Huggett-anchored); the T_emit_gate
raise is a tuning-session item with its stated accuracy cost (the bounded
sub-gate sky loss); the hysteresis phrase is DELETED (single threshold + the
boundary gate). **Ships-O2 scope ruling recorded:** vents and reservoirs are
map design, not fire law — the reason 2b is the ONLY supply term.
**Decision 4 reframed:** room-feel targets are DAMAGE-ONSET curves.
**Decision 5, both halves:** units carry a felt temperature from their
surroundings including air; equipment sets safe ranges — logged to the
unit-system design queue (priority-ledger §2 riders). **Spread band:**
30-60 s governs (the req-6 margin said 15-30; the §5 answer supersedes);
"hot fires spread faster" is an expectation, not a gate. **Flicker** is a
named SCALABLE feel dial (the coupling gains) at P-R5. **The smoke
slow-tick compounding-ray idea** lives in §7's smoke-radiation entry.
**P11 corrected:** the dem_acc stale-debt fix is ALREADY IN-TREE; remaining
shipped fixes = the H_bed int32 clamp path + deleting the mt19937
distribution. **Touchpoint table (honest):** the sizing call, wind anchors,
eps/occlusion feel, room-feel curves, the P-R5 session, the ember review,
and ~5 play tests.

### v5.4 — Verification and immediate work

A narrow round-3.5 agent re-derives v5.1 (the mask split, all three refunds,
clamp bookkeeping) against the four limits and the two-temperature gate
BEFORE P-F1a spawns. P-F4a and the P11 mechanical fixes gate on nothing
above and are buildable immediately.


---

## v6 — THE MINIMAL BOOKS (2026-08-02, after the round-3.5 verification;
## SUPERSEDES v5.1 ENTIRELY. The sink, the credit, and all three refunds are
## DELETED. Verdict pending round-3.6 verification.)

Round-3.5 (`fire_realism_critiques_round3_2026-08-02.md` + the 3.5 report in
the session record) found v5.1 unclosed (sink outside the ledger identity;
opaque refund minting energy via a missing (1-a_r); range refund abolishing
open-air loss) and, decisively, showed via the per-direction budget that the
sink + credit + refunds telescope into a far simpler object. v6 is that
object. Five compensating terms become four rules:

### v6.1 — The law

Emission rays are cast by emitters only (burning OR thermal_solid with
T >= T_emit_gate — unchanged). Per direction d with weight w (Sum w = 1),
marching with transmittance tau as today:

1. **Non-emitter absorbing cell r:** the ORIGINAL antisymmetric pair, ONE
   integer applied +/-: `x = a_s*a_r*tau*w*(E_emit(s) - E0[T_r])`;
   `rad_net[r] += x; rad_net[s] -= x`. (The S1 shared-truncated-integer
   idiom RETURNS; conservation is exact with no ledger involvement.)
2. **Emitter absorbing cell r:** one-way potential-vs-ambient, ONE integer
   moved: `x = a_s*a_r*tau*w*(E_emit(s) - E0[0])`; `rad_net[r] += x;
   rad_net[s] -= x`. (r's own emission books on r's own cast — kills the
   mutual-emitter 2x; two equal-T emitters net zero at 1x rate.)
3. **Contact faces are radiation-inert:** a ray entering a solid-solid
   CONTACT face (the marched cell is solid AND shares that face with the
   previous solid cell — conducting or not) terminates with NO deposit and
   NO charge; contact is conduction's domain (Erik ruling 3; the zero-kappa
   contact case is a named negligible lump). A fully-enclosed tile thus
   exchanges nothing radiatively by construction — the interior-wall mask
   becomes a pure optimization with identical books.
4. **The sky term — the ONLY ledger entry:** when a ray genuinely escapes
   (grid edge, or reach-termination in the open), the emitter is charged the
   escaping residual: `sky = a_s*tau_end*w*(E_emit(s) - E0[0])`;
   `rad_net[s] -= sky; rad_amb += sky`. THE RANGE FLOOR makes "genuinely
   escapes" well-defined: emission rays use a fixed RADIATION_RANGE >= the
   design maximum room span (a global constant; the per-intensity range
   formula does not apply to emission rays — reach is geometry, air cells
   cost no deposit work). Indoors, opacity or contact terminates first;
   reach-termination therefore IS the open field.

There is no sink, no credit, no refund. Self-cell: excluded from all
deposits as before (pair and one-way; distance 0 books nothing).
phi/flame_lift ride E_emit as in v5; the source-cell exclusion covers
everything.

### v6.2 — The limits (exact, re-derived; round-3.6 verifies)

(a) Lone emitter, open air: every ray escapes; loss = a_s*(E_s - E0). Full
grey-body. (b) Ambient enclosing walls: pairs telescope to the SAME
a_s*(E_s - E0) — Erik's equivalence holds exactly, with zero additional
machinery, at any room size >= handled by the range floor. (c) Two equal-T
emitters: rule 2 both ways nets zero; at DT the exchange is 1x. Sub-gate
tiles cast nothing and pay nothing — the below-gate regime is "does not
radiate" (the original A1.8 semantics), so there is no unrefunded-sink trap,
no interior heat trap, and the gate-crossing step is the PHYSICAL onset of
emission, bounded by E0(T_gate) - E0(0) (~1.7 game/tick at gate 180), listed
as the known lump the tuning session may move (Erik's gate-raise lean).
(d) Hot emitter, cold non-emitter: receiver gains the pair; the emitter's
net loss per covered direction is the pair itself. Sealed equal-T room above
the gate: all terminations are opaque or contact, no sky is charged, incomes
equal outflows to reciprocity; net zero per tile.

### v6.3 — Clamps, gates, ledger (round-3.5's remaining items)

- Every transfer x is clamped by the SHARED symmetric pair budget
  (rad_pair_budget, signed, per term); the cumulative per-emitter credit
  budget of v5 is DELETED (it was order-dependent on CUDA). The aggregate
  fold clamp + Pass-1 rails stay as counted diagnostics and must be ASSERTED
  INERT in every gate scenario (a counter hit in a gate run = red).
- Ledger identity: `Sum(rad_net) + rad_amb == 0` holds structurally (pairs
  move integers; sky is the lone +/- ledger pair). Gate (ii) tests exactly
  the sky bookings and int32 wrap.
- Gate (iii) becomes the NET test: on a fire-free uniform equal-T grid above
  the gate, every tile's rad_net == 0 exactly (rule-2 symmetry) — and on the
  same grid below the gate, trivially 0 (nobody casts).
- Gate (iv): (a) sealed equal-T box below the gate: all rad_net == 0
  (nothing radiates); (b) sealed two-temperature box above the gate:
  exchange = 1x one-way law, continuous in each tile's own T; (c) the
  crossing step at the gate is MEASURED and reported against its derived
  bound (not gated to zero — it is the emission onset).
- Gate (v): sealed equal-T room WIDER than the old max_range: zero net per
  tile (exercises the range floor); open-field lone emitter: rad_amb equals
  the full grey-body rate (exercises the sky term).
- `range_base`/`range_per_intensity` return fully to render/legacy duty;
  RADIATION_RANGE is a new stability-class constant (not a feel dial),
  sized to the maximum design room span, cost statement: air cells cost a
  march step and no deposit.

### v6.4 — Status

v5.2 (F-O2b) and v5.3 (rulings fold) stand unchanged. v5.1 is void. The
round-3.6 narrow verifier re-derives v6.1-v6.2 before P-F1a spawns; P-F4a
and P-O2b are unaffected by any of this and proceed.

---

## v7 — THE SYMMETRIZED BOOKS (2026-08-02, after round-3.6; SUPERSEDES v6's
## rule 2 and amends rules 3-4. Verdict pending round-3.7.)

Round-3.6 certified v6's conservation, contact well-definedness, limits (a)
and (d) incl. the negative pair, and the sky term's determinism — and broke
rule 2 decisively: the |dT|-based clamp annihilates a potential-vs-ambient
term below dT ~ 76 game (mutual-emitter feeding zero across the whole
operating band, ledger green), and a gap-blind term has no valid stability
rail and depends on a ray-fan reciprocity the D4 record proves absent
(cold-to-hot positive feedback possible). Its diagnosis: "rule 1's
antisymmetry was doing three jobs — conservation, second-law safety, and
making the flux limiter a valid rail." v7 keeps rule 1's form for EVERYTHING:

### v7.1 — The law (delta from v6)

- **Rule 2 (replaced): mutual emitters use the SAME gap-signed antisymmetric
  pair as rule 1, at HALF weight:** `x = mul_q16(HALF_Q, a_s*a_r*tau*w*
  (E_emit(s) - E_emit(r)))`; `rad_net[r] += x; rad_net[s] -= x` (one integer,
  +/-). The two casts of a mutual pair sum to exactly 1x the antisymmetric
  exchange (no double-count); the term is gap-signed (the shared |dT| clamp
  is valid and inert in-band; heat can never flow cold-to-hot regardless of
  fan asymmetry — worst case is a bounded RATE error, second-law safe); two
  equal-T emitters exchange exactly 0 structurally. THE CROSSING BECOMES
  CONTINUOUS in the exchange: below the gate s casts the full pair; above,
  each casts half — the s<->r exchange rate is identical across r's crossing
  (what changes at the gate is only r beginning to pay its OWN other
  directions and sky — the physical onset of emission). Emitter-vs-emitter
  uses E_emit on BOTH ends (flame lift both sides). Note the half-weight
  branch keys on the SAME mask plane; membership stays burning OR >= gate.
- **Rule 3 (amended scope): contact directions are NON-PARTICIPATING, and
  that is the stated semantics, not a leak.** A ray that enters a solid-solid
  contact face terminates; the direction's residual is charged to NOBODY —
  the emitter's radiative loss simply excludes directions occluded by
  contact and by interior assemblies (they conduct instead; the zero-kappa
  contact case is the named negligible lump). Stacked absorbers/flush glass:
  the interior of an assembly does not radiate — intended, stated. The
  telescoping identity for the equivalence limit is over NON-CONTACT
  directions; open-air vs ambient-walls compares like-for-like and stays
  exact. (Round-3.6 MAJOR-1 resolved by declaration, with the semantics
  Erik's conduction ruling already implies.)
- **Rule 4 (amended): RADIATION_RANGE >= the GRID DIAGONAL** — reach-
  termination can then never precede the grid edge, "genuinely escapes" ==
  "left the world", map-independent; the corridor leak is structurally
  impossible (round-3.6 MAJOR-2). COST: mandatory PURE-RADIATION FAST PATH —
  emission rays skip the RGB/light and gas-optics work entirely (they need
  only heat_atten tau + the integer terms); gate (g) re-measured with the
  fast path + long rays (round-3.6 MAJOR-3).

### v7.2 — Fold and rails

The rad_net -> T fold keeps shr_round0 with NO carry: the discarded
sub-2^his remainder is a bounded, sign-symmetric quantization floor (< 1
T-LSB/tick), ACCEPTED and named (a far cold wall receiving < 1 LSB of flux
does not creep — fine for gameplay; the D1-style carry would need a
persistent synced plane and forfeit the golden freeze). LOW rail at 0 with a
counter, justified by the budget argument (all terms clamped to |dT|/16-
shares; aggregate bounded); rails asserted inert in gate scenarios.

### v7.3 — Gates (round-3.6's coverage repairs)

(ii) ledger: sky bookings + wrap, as before. (iii) NET test on an
AIR-SEPARATED equal-T emitter lattice (rule-2 symmetry actually exercised;
contiguous-solid grids are vacuous under rule 3) — every rad_net == 0
exactly. (iv)(b) two-temperature box swept over dT INCLUDING small dT (5,
10, 20, 40 game — the operating band; catches any clamp bite), exchange =
1x the pair law, continuous across each tile's own crossing, swept BOTH
directions. **(iv)(e) THE EQUIVALENCE GATE — the arc's headline, previously
ungated (round-3.6 BLOCKER-3): one emitter's measured net loss open-field
vs centred in a sealed ambient-T room (walls pinned T=0, a=1.0 single-layer
geometry where telescoping is exact) — equal to (n+1)/2-count tolerance;
repeated at a=0.5 walls with the derived geometric-defect tolerance.**
(v) as v6 (room wider than the OLD max_range now trivially covered; keep as
regression). Negative-pair case (hot sub-gate solid heats a cooler emitter)
gets one directed scenario. v6.2(c)'s "sub-gate tiles pay nothing" is
corrected: a sub-gate solid pays/receives via rule 1 whenever an emitter
sees it — what it does not do is CAST.

### v7.4 — Status

v5.2 (F-O2b) and v5.3 (rulings fold) unchanged. v6.1 rules 1 and self-cell/
E_emit conventions stand; this block replaces rule 2 and amends 3-4.
Round-3.7 verifies v7.1-v7.3 before P-F1a spawns. P-O2b is unaffected and
may start once P-F4b's sweep tooling question (same patch family) is set.

---

## v7.1 — CLOSURE EDITS (2026-08-02, round-3.7's prescribed fixes, folded
## verbatim. THE BOOKS ARE CLOSED FOR BUILD: conservation, ledger identity
## and second-law safety CERTIFIED (round-3.7); the edits below are that
## verifier's own prescriptions, folded without alteration.)

1. (M1) The gap-signedness / clamp-validity claims of v7.1 are SCOPED to
   phi == 1, flame_lift == 0 — which is exactly what P-F1a ships. Any later
   patch that turns on flame lift MUST first define the per-end T_eff (the
   E_emit lookup argument) and size the budget from |dT_eff|; until then
   lift stays off. 2. (M2) The mutual half-branch caps at HALF the shared
   budget (RAD_LIM_SHIFT + 1 in that branch), clamp-after-halve — the rail
   stays true by construction with both ends casting. 3. (M3) Gate (iv)(b)
   measures over a whole multiple of fire_ray_count ticks; the continuity
   claim is restated: equal in expectation over the D4 rotation period,
   reciprocity-limited per pair per tick, worst case a factor-2 rate step
   with no sign change. 4. (M4) D3's rad_flux sensor KEEPS THE OLD REACH: a
   deterministic distance guard (damage_range = the legacy range formula) on
   the rad_flux write only — books untouched, unit radiant damage unchanged;
   far-field bursts never ship. 5. (M5) Gate (iv)(e) geometry PINNED:
   air-separated walls (no wall tile face-adjacent to the emitter), a = 1.0
   single-layer grid-border-backed walls, tolerance ZERO (the pair value is
   bit-identical to the sky charge there), E0[0] is literally e_table[0];
   the a = 0.5 variant runs with a contact-termination counter and a
   tolerance derived from the measured contact-direction count. 6. (M6)
   Gate (g) re-measured on the largest shipping level (128x256) open-field
   firestorm WITH the pure-radiation fast path; the acceptance carries an
   emitter-count statement (the caster set is temperature-defined and
   unbounded — a cap/LOD policy, deterministic, is part of P-F1a's spec if
   the measurement demands one). Fire's visible light becomes a second
   short-range cast (cheap, golden-neutral) — the fast path does not carry
   RGB. 7. (m1) The half factor is x0.5f INSIDE the pinned float fold,
   before the single quantize (one rounding boundary, sign-symmetric;
   recombination residual <= 1 count, annihilated by the fold) — NOT
   mul_q16(HALF_Q, .). Pinned in both backends. 8. (m2) The sky term is
   clamped by rad_pair_budget(|T_s|, his_s) — the ambient counterparty is
   the T = 0 partner. 9. (m3) Gate (ii)'s ledger identity is evaluated
   PRE-FOLD (the fold discards the sub-2^his remainder, bias toward zero in
   magnitude = systematic slight under-transfer, never a mint). 10. (m4)
   CPU-CUDA tol-0 step+resident is restated as gate (ix) for the half
   branch, the sky term, and the fast path. 11. (m5) The heat_cull residual
   (<= 1% of a direction share charged to nobody at cull termination;
   under-cooling, safe direction) is named; gate (v)'s open-field grey-body
   tolerance is set BELOW it. 12. (m6) Rule 3 sentence added: a
   wall-adjacent crate is EXPECTED to lose less radiatively than the
   open-field crate (its contact face conducts instead); Erik's equivalence
   compares open field vs AIR-SEPARATED ambient scenery. 13. (3.7 §5b) The
   emitter-mask compare is a single integer threshold against the same
   temperature snapshot on both backends; one boundary-tile scenario pins
   a tile at exactly quantize(T_emit_gate). 14. (3.7 §5c) One extreme-gap
   scenario asserts monotone approach when the rail FIRES (not merely that
   it did not). 15. (3.7 §4) The fold dead-zone acceptance is a GAMEPLAY
   statement, named for Erik at P-R5: a tile receiving under 1 T-LSB per
   tick of flux does not creep toward ignition; D4's burst delivery is why
   distant exposure still accumulates in practice.

**STATUS: the radiation books (v6.1 rules 1/3/4 as amended by v7 + these
edits) are the BUILD SPEC for P-F1a. No further paper rounds; P-F1a's
sharpened gate set is the empirical backstop. P-O2b proceeds in parallel
(independent subsystem, spec v5.2).**
