# Fire session #12 — R3 report + full 3c state (2026-09-06)

> **Purpose**: the extensive report Erik asked for, written to be
> self-contained for a FRESH analysis session (no prior context assumed).
> Covers what was built, what was measured, what it means, what is still
> open, and — explicitly — what is VERIFIED versus what is INFERRED.
>
> Companion docs (read in this order if you want the full chain):
> `fire_mechanics_inventory_2026-08-31.md` (what the engine does) →
> `fire_phase3a_measurements_2026-09-01.md` (the July re-measurement) →
> `fire_3c_prebench_2026-09-01.md` (O2-wall + cluster benches) →
> `fire_3c_design_2026-09-01.md` (the live ruling doc, R1/R2/R3) →
> THIS FILE.

---

## 1. Branch state (facts)

Branch `fire-12`. Pushed through `1ea4431` (the R3 ruling text).
**`2b9f7f3` (the R3 implementation) is COMMITTED LOCALLY AND NOT PUSHED**
— deliberately, pending the anchor decision in §6. Nothing else is
uncommitted.

Session sequence, all on `fire-12`:

| Commit | What |
|---|---|
| `f1f7ecd` | G12 — one temperature map (K = 293 + T_game). Erik-blessed. |
| `658bda8` | Tile-inspector HUD on by default |
| `3155b86` | #59 — G-key no longer toggles sRGB (was double-bound) |
| `7e34aaf` `f0b6a7e` | Phase 2 — full HUD field set + `levels/fire_tuning` |
| `a9d4681` | Phase 3a — July re-measurement memo |
| `f09e49f` `c819337` | 3c pre-benches (no O2 wall; cluster coupling) |
| `32bb03c` | **R1** — o2f renormalized to ambient |
| `0c1a370` | **R2** + the T\* re-derivation |
| `1ea4431` | **R3** ruling text |
| `2b9f7f3` | **R3 implementation (LOCAL ONLY — under review)** |

---

## 2. The three rulings, in one paragraph each

**R1 (o2f renormalized to ambient).** The sustain-side oxygen factor was
`(X − 0.13)/0.87` — normalized to 1 at *pure oxygen*, so ordinary air
(X = 0.21) read **0.092**, and every other fire dial was silently sized to
compensate. Now `(X − 0.13)/(0.21 − 0.13)`, capped at 5 for enrichment:
1.0 at ambient. The extinction foot (0.13) was deliberately left untouched
— it is the flicker/death dial. The demand side keeps the RAW factor
(consumption physics unchanged). `I_cap_per_avail` was re-sized 14 → 0.95
by closed form against the measured plateau. Landed, verified, pushed.

**R2 (both loss channels stay) + the T\* re-derivation.** `cool_shift` is
linear in T (a Newton-cooling/convection proxy); radiation is T⁴. Both
stay. The config's plateau formula overshot reality ~110× because it
omitted radiation entirely. Re-derived: **the plateau is a
Stefan–Boltzmann equilibrium — a fourth-root law.** At the measured
plateau, radiation carries **≈99.6%** of an emitter's losses and
cool_shift **≈0.4%** (crossover at T ≈ 10 game). Independently
cross-validated by 3a's M5 sweep (predicted K-ratio 1.50, measured 1.43).
Consequence: reaching G1's 1300 K needs deposit ×≈9, and **H_bed is the
lever** — `k_grow` and `cool_shift` are near-irrelevant to emitter
temperature.

**R3 (hot-burns-faster).** Erik's catch: `hot` clamps at 1, so above
≈T_ext+span extra heat buys ZERO extra burn rate. R3 adds
`hotf = clamp((T − T_ext[mat])/span, 0, cap)` — the same ramp allowed to
keep climbing — applied to the O2 DEMAND and to the DESTRUCTION rate.
Implemented; **measurements below say the calibration anchor was wrong.**

---

## 3. What R3 actually implements (verified by reading the code)

**Law**: `hotf = clamp((T − fire_T_ext[mat]) / fire_T_span, 0, hotf_cap)`,
`hotf_cap = 10.0` (new dial). The clamped `hot` is UNCHANGED as the
sustain gate in the I-ODE — `hotf` is a separate, uncapped-at-1 read of
the identical ramp.

**Site 1 — O2 demand**, `cpp/src/combustion.cpp:571-574` (CUDA twin
`cuda_combustion.cu:298`), verified by reading:
```cpp
const q16 hotf_i = clamp0cap_q(recip_mul(Tsnap[i] - T_ext_i, recip_T_span), hotf_cap_q);
const q16 o2f_hotf = mul_q16(o2f_j, hotf_i);
```
`hotf` is folded into `o2f_j` with ONE narrowing multiply *before* the
wide product forms. This is load-bearing for determinism: as a naive
fourth raw factor the product would bound at `hotf_cap·2^64` and overflow
int64; pre-narrowing bounds it at `hotf_cap·2^48` (≈2^51.3 at cap 10). Cost
is ≤1 LSB on a quantity that is itself a lossy per-tick sensor read. T
source is `Tsnap` (the pass-entry snapshot), preserving "a source cannot
heat AND ignite a neighbour in the same tick".

**Site 2 — destruction**, `cpp/src/fire_simulation.cpp:352-362` (CUDA twin
`cuda_fire.cu:303`):
```cpp
const q16 wd = mul_q16(mul_q16(wall_damage_q, dt_q), hotf);
wall_hp[i] -= narrow_round(mul_wide(wd, fire[i]));
if (wall_hp[i] <= 0 && flammable[i] && is_wall[i]) { destroyed.push_back(...); }
```
**Note the asymmetry (important for §6.2)**: the *destroy* decision is
already flammable-gated; the *hp depletion* above it is NOT. Fire drains
hp from non-flammable tiles, which then can never be destroyed.

**Fuel drain needs no separate term** — `fuel_cost = fuel_per_o2 ·
O2_drawn` already, so drawing more oxygen automatically eats more fuel.
Verified: `combustion.cpp:751`.

**Neutral-landing re-size** (the decision under review): both rate dials
divided by `f_ref = 2.0717` = `hotf` at the measured plateau T = 452.9:
`burn_rate` 0.02 → 0.00965, `wall_damage` 0.03 → 0.01448.

---

## 4. Measurements (all numbers from `tests/_phase3a_artifacts/`)

### 4.1 M1 — the reference single furniture crate, open sky-fed room

| Metric | 3a baseline | after R3 | direction |
|---|---|---|---|
| Peak intensity | 0.696 @ 122 s | **0.849 @ 13.1 s** | higher, ~9× sooner |
| Time to 90% of peak | 80.3 s | 9.9 s | much faster |
| Plateau intensity | — | 0.484 | — |
| Plateau temperature | ~738 K (near-peak) | **636.7 K** (343.7 game) | **colder** |
| Death | 1645 s (27.4 min) | 1139 s (19.0 min) | sooner |
| Fuel unburned at death | 26.7% | **44.1%** | worse |
| Cause of death | heat/O2 collapse | heat/O2 collapse | unchanged |

### 4.2 b1 open control — infinite fuel (wall_hp pinned), sky-fed

| Metric | post-R1 | after R3 | target |
|---|---|---|---|
| Plateau intensity | 0.7112 | 0.7170 | preserved (+0.8%) |
| Plateau temperature | 452.9 game | 451.7 game (744.7 K) | preserved (−0.3%) |
| Died? | no (60 min cap) | no (cap) | — |

**The re-size did exactly what it was calibrated to do — in the one
scenario it was calibrated on.**

### 4.3 b1 sealed — infinite fuel, sealed 12×12 chamber

| Metric | pre-R1 | post-R1 | after R3 |
|---|---|---|---|
| Death time | 1772 s | 2038 s | **590 s** |
| X at death | 0.176 | 0.166 | **0.1857** |
| Proximate cause | T-gate | T-gate | T-gate |
| T at death | — | — | 3.7 game (296.7 K) |
| Plateau T | — | — | 183.9 game (476.9 K) |

The ruling predicted X_death would fall toward the 0.13 oxygen wall.
**It rose, and the fire died 3.5× sooner.** Reported as measured.

---

## 5. Diagnosis — why it went the wrong way (inference, but well-supported)

The mechanism is a chain, and every link is checkable:

1. A fire is **born at its ignition temperature** (~280 game for
   furniture), where `hotf = (280 − 80)/180 = 1.111`.
2. But both rate dials were divided by `f_ref = 2.0717` (the factor at
   *plateau* temperature). So a young fire draws
   **1.111 / 2.0717 = 54% of the oxygen it drew before R3.**
3. Less O2 claimed → less H_bed heat deposited (deposit ∝ O2 claimed) →
   the tile runs cooler → `hotf` stays low → **less deposit still.** The
   feedback we installed runs *downward* from a cold start. The fire never
   reaches the temperature where the acceleration would pay for itself.
4. Simultaneously, less O2 drawn → less fuel drained (`fuel_cost ∝ O2`) →
   `F = wall_hp/hp` stays near 1 → `avail = F·o2f` stays high → **the
   intensity ODE climbs faster and peaks higher.**

That is the measured signature exactly: **higher intensity, lower
temperature, more fuel left over.** We accidentally built "burn bright and
cold" — the wrong side of both G1 (want hotter) and G4 (want lower I).

**Why the open infinite-fuel control was preserved anyway**: there `F ≡ 1`
by construction, so the fire is strong enough to hold the reference
temperature — the exact point the re-size was anchored to. Fuel-limited
fires slide to a lower fixed point. The anchor was calibrated on the one
scenario that cannot show the flaw. That is a methodological lesson worth
keeping: **anchor a calibration on the regime you intend to ship, not on
the most convenient measurement.**

---

## 6. Open decisions

### 6.1 The anchor — and why per-material ignition is NOT a problem

Erik's concern: "anchor to ignition perhaps doesn't work, since every
material has its own ignition temperature."

**It works, and the reason is a structural identity.** `fire_T_ext` is not
independent — it is *derived*:
```
fire_T_ext[mat] = ignition_temp[mat] − ignition_to_ext_delta        (Δ = 200)
```
So evaluating `hotf` at each material's OWN ignition temperature:
```
hotf(ignition_temp[mat]) = (ign[mat] − (ign[mat] − Δ)) / span
                         = Δ / span = 200 / 180 = 1.1111…
```
The material cancels. **Every flammable material has exactly the same
`hotf` at its own ignition point**, so one global anchor is exact for all
of them, today and for any material added later (as long as Δ stays
global, which is itself a structural rule adopted at P-R3). Verified
against the bench: furniture's `fire_T_ext` = 80 with ignition 280. ✔

Proposed re-anchor: `f_ref = Δ/span = 1.1111` instead of 2.0717, i.e.
`burn_rate` 0.02 → **0.018**, `wall_damage` 0.03 → **0.027**. Effect: all
rates ×1.86 versus the current R3 landing.

Behaviour this buys, and it is exactly Erik's stated want ("slow start,
accelerating as it gets hotter"):
- At ignition: **identical to pre-R3**. The bootstrap is preserved
  exactly — no starved young fire.
- At today's plateau: deposit **×1.86**.
- At the cap: deposit **×9**.

**Predicted plateau (INFERENCE, to be measured)**: solving the R2
fourth-root balance with the feedback included —
`hotf(T)/1.1111 = (K⁴ − K_amb⁴)/(K₀⁴ − K_amb⁴)`, K₀ = 744.7 K — gives an
equilibrium near **T ≈ 685 game, K ≈ 978** for the open infinite-fuel
control, up from 745 K. (A naive one-step estimate ignoring the feedback
gives 868 K; the compounding is what carries it further.) Caveat: this
holds X fixed, and heavier drawing will depress X and hence o2f, so treat
978 K as an optimistic bound. Either way it is real progress toward G1's
1300 K *before* Phase 4 touches H_bed.

### 6.2 Erik's ruling (2026-09-06): fire may not destroy non-flammable tiles

Recorded as **R4**. Current code: the destroy decision IS gated
(`flammable[i] && is_wall[i]`), but the `wall_hp` depletion above it is
not — so fire quietly drains hp from non-flammable tiles that can never
burn. Implementing R4 = gate the depletion too.

Two consequences worth noting:
- It is a small correctness fix independent of R3.
- **It also removes the golden movement** described in §6.3, because the
  canonical scenario's fire sits on non-flammable tiles.

### 6.3 The golden, and the ghost fire

Erik asked whether ghost fires were retired. **Partly.** The canonical A/B
scenario still seeds fire at (8,8)/(8,9) on AIR tiles (material 0,
`flammable.sum() == 0`). What P-R4 retired in 2026-08-01 was its *heat
observable* — under Kirchhoff a body that cannot absorb cannot emit, so
its radiation contribution became correctly zero. **The seeded fire tiles
themselves are still there**, and they still flow through the
destruction/`wall_hp` loop, which is why R3 moved the golden
(`54f21b36…` → `d645b939…`).

All 12 failing `test_cuda_*` files fail on **this one shared golden hash
only** — I verified their own CPU↔GPU bit-identity legs pass (e.g.
`test_cuda_trace_smoke`: "PART 1 — isolated GPU vs CPU … " passes, then
the golden comparison fails). This is the documented cascade pattern, not
GPU divergence. **R3's CUDA parity is genuinely green.**

If R4 (§6.2) lands with the re-anchor, the golden should stop moving
altogether and no re-baseline is needed — the cleanest outcome.

### 6.4 Two different "ramps" — a measurement gap worth closing

Erik's want is "the increase in **temp** starts slow, then accelerates".
Everything the benches call "ramp" (G2's 30–120 s target, the 80.3 s
figure) is the **intensity** ramp — time for `I` to reach 90% of peak.
These are different curves with different levers:
- **Intensity ramp** ← `k_grow` (the explicit TEMPO dial), `I_cap`.
- **Temperature ramp** ← deposit vs T⁴ losses, i.e. H_bed and now `hotf`.

R3 makes the *temperature* curve accelerate (the desired shape) while the
*intensity* curve is governed elsewhere — which is why R3 could sharpen
the I spike and cool the fire at the same time. **Recommendation: the
benches should report both ramps separately from here on**; G2's target
should be restated against whichever curve Erik actually cares about.

---

## 7. Verified vs inferred (so the next session can trust this correctly)

**Verified by direct execution or by reading the shipped code:**
- All measured numbers in §4 (raw CSV/JSON in `tests/_phase3a_artifacts/`).
- Both `hotf` code sites and the overflow-avoidance fold (§3, read).
- The destruction/depletion gating asymmetry (§6.2, read).
- CUDA failures are golden-hash-only, bit-identity legs pass (§6.3, ran).
- Full suite: 2342 passed; failures = 3 known reds + the golden cascade.
- Both builds (CPU, CUDA) exit 0.
- The `Δ/span` material-independence identity (§6.1, algebra + bench
  cross-check on furniture).

**Inferred (reasoned, not yet measured):**
- The §5 causal chain. Every link is individually supported, but the chain
  as a whole is an explanation, not a measurement. The decisive test is
  the re-anchor rerun: if §5 is right, ignition-anchoring should raise the
  plateau and *reduce* the intensity spike (more O2 early → more fuel
  drained → `F` falls → lower `I`).
- The ≈978 K plateau prediction (§6.1) — explicitly an optimistic bound.

**Not attempted this session**: exposure-integral ignition (G8),
ember/auto-reignite/wind-strip hysteresis (G6/G7/G11), the o2f vacuum
amendment (#7), and replacing the config's stale T\* comment block with
the corrected fourth-root law.

---

## 8. Reproduction

```
# builds (this box, Home Desktop)
cmd /c cpp\build_cpu_home.bat
cmd /c cpp\build_cuda.bat

# suite (CPU only, if the GPU is busy)
C:/Users/steen/anaconda3/python.exe -m pytest tests -q -k "not cuda"

# the benches (raw output -> tests/_phase3a_artifacts/, untracked)
C:/Users/steen/anaconda3/python.exe tests/_phase3a_driver.py --b1 --m1
```
`--b1` runs both infinite-fuel legs (sealed + open control); `--m1` the
reference single-crate run. Summaries land as `*_summary.json` beside the
per-tick CSVs.

---

## 9. Recommended next step (one line)

Re-anchor to `f_ref = Δ/span = 1.1111`, land R4's non-flammable gate in
the same patch, rerun `--b1 --m1`, and expect: bootstrap unchanged,
plateau up (~900+ K), intensity spike *down*, golden still.
