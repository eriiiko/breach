# Fire 3c design — rulings + patch specs (session #12, live doc)

> **Status: LIVE SESSION DOC (Erik + Claude, 2026-09-01 evening).** Rulings
> land here as they are made; each becomes a patch spec. Inputs:
> `fire_3c_design_brief_2026-09-01.md`, `fire_phase3a_measurements_*.md`,
> `fire_3c_prebench_*.md`.

## Ruling R1 (Erik, 2026-09-01): o2f renormalized to ambient — LOCKED

**The law**: sustain-side o2f becomes
`o2f = clamp((X − o2_frac_ext) / (o2_frac_amb − o2_frac_ext), 0, o2f_cap)`
with `o2_frac_ext = 0.13` (foot UNTOUCHED — it is the flicker/death dial),
`o2_frac_amb = 0.21`, and **`o2f_cap = 5.0` (NEW dial — the enrichment
flare ceiling; Erik's choice, raw line would reach 10.875 at pure O2)**.
So o2f = 1.0 at normal air (was 0.092 — every dial was compensating).

**Why (the pre-bench diagnosis)**: under pure-O2 normalization, a mild
local X dip (0.21→0.165) halves an already-tiny o2f, I_eq collapses,
heat deposit (∝ I·o2f_demand) collapses, T falls through fire_T_ext,
hot→0, I snaps to 0 — death by cold with O2 far above the gate
(measured: X_death 0.176 and rising). Renormalized, avail ≈ 1 at ambient
keeps I high at modest depletion → deposit holds T up → fires survive
toward the foot and die NEAR the gate. Bonus: k_die=0.008's logistic wall
lands at X ≈ 0.131 under this scaling — the config comment's own claim
("just above o2_frac_ext") becomes true.

**Scope decisions (all part of R1):**
1. SUSTAIN side only: the I-ODE's o2f (fire_simulation.cpp + CUDA twin).
   **DEMAND side stays raw** — combustion.cpp's o2f_j (O2 drawn,
   consumption rate, H_bed deposit per unit I) is UNCHANGED, so sealed
   rooms deplete at today's honest rate. Two roles, two shapes: "how well
   it thrives" (renormalized) vs "how fast it drinks" (raw).
2. Ignition gates unchanged (X > o2_frac_ext, both paths).
3. **Die-term sign trap fixed**: avail can now exceed 1 (enrichment), so
   `die = k_die·(1 − avail·hot)·I` would go NEGATIVE (anti-death).
   Becomes `die = k_die·max(0, 1 − avail·hot)·I` (+ wind term unchanged).
   Enrichment boosts fires through grow/I_cap only.
4. `I_cap_per_avail` re-sized 14.0 → **0.95** — closed-form against the
   MEASURED plateau availability (a=0.7935, not the naive a≈1):
   c = 0.75/(a − r(1−a)) ≈ 0.9492; derivation now lives in config.toml's
   own comment. One-time mechanical re-size; Phase 4 owns the taste pass
   (Erik's "hotter at lower I" = H_bed work, deliberately NOT this patch).
5. Config: `o2_frac_amb` becomes live (was fallback); `o2_frac_full = 1.0`
   retired with tombstone; `o2f_cap = 5.0` new; k_die comment's death-wall
   claim re-derived (now true).
6. Golden: verified byte-UNCHANGED (the canonical scenario's ghost fire
   never touches the sustain logistic — no re-baseline was legitimate;
   the fire-less golden thus served as a clean no-side-effects proof).
   Real-fire golden design deferred to #13 (ruling 2026-09-01, comment
   there).

**AS LANDED (`32bb03c`, 2026-09-01/02, verification bench results):**
- Sealed infinite-fuel: X_death 0.176 → **0.166** (toward the 0.13 gate),
  life 29.5 → 34.0 min, proximate cause STILL T-gate → the remaining gap
  is the hot-burns-faster item's territory (+ foot shape if wanted).
- Open control: I 0.751 → 0.711 (−5.3%), T 460.5 → 452.9 game (−1.6%) —
  plateau preserved within tolerance. Closed-form I_eq now predicts the
  measured plateau to 3 decimals (the config's formulas are TRUE again).
- Suite 2330 green CPU-only (`-k "not cuda"`), 3 known reds only.

## Ruling R3 (Erik, 2026-09-06): hot-burns-faster — LOCKED

Erik's saturation catch: `hot` clamps at 1, so above ~T_ext+span extra
heat buys ZERO extra burn rate today. R3 makes burn rate temperature-
dependent, with **O2 deprivation as the structural cap** (Erik): sealed →
the reservoir caps then kills it; open → the supply RATE caps it
(ventilation-limited burning, the real-fire regime).

**The law (shape B — "extended hot", Erik's ruling):**
```
hotf(T) = clamp( (T − fire_T_ext[mat]) / fire_T_span , 0 , hotf_cap )
hotf_cap = 10.0                      (NEW dial, pairs with R1's o2f_cap)
```
`hot` (clamped at 1) STAYS the sustain gate in the I-ODE; `hotf` is the
same ramp allowed to keep climbing, read only by the two rate sites below.

**WHY the linear shape, not Arrhenius** (the load-bearing argument): R2
established losses ∝ T⁴ (radiation 99.6%). A LINEAR hotf can never
outrun T⁴, so an equilibrium always exists — the system is self-limiting
BY CONSTRUCTION and Erik's O2 cap is a second, gameplay-flavoured ceiling
on an already-stable system. An exponential (Arrhenius) hotf eventually
beats T⁴, making stability depend on O2 running out first — fragile
exactly where oxygen is generous (breach airflow, enriched rooms).
Arrhenius is the more "correct" chemistry; it is the wrong numerics here.

**Two application sites (both, per Erik's ruling 3):**
1. **Demand** (`combustion.cpp`): `demand_k = burn_cap·I_k·o2f_j·hotf_k·
   W_hop·w_path` — in the PINNED left-fold order, Q16, overflow-checked
   (the fold gains a factor ≤ 10). T source = `Tsnap` (pass-entry
   snapshot — preserves "a source can't heat AND ignite the same tick").
   Fuel DRAIN needs no separate term: `fuel_cost ∝ O2 drawn` already.
2. **Destruction** (`fire_simulation.cpp`): `wall_damage·dt·I` becomes
   `wall_damage·dt·I·hotf`. (The never-destroys invariant is unaffected —
   it governs the combustion drain path, which still floors at
   FUEL_FLOOR; destruction remains this one site.)

**Neutral landing (Erik's ruling 1) — BOTH dials re-size, same factor.**
Reference = the measured post-R1 open-control plateau T = 452.9 game;
furniture `fire_T_ext` = 280−200 = 80, `fire_T_span` = 180 →
`f_ref = (452.9−80)/180 = 2.0717`. Therefore:
```
burn_rate    0.02  → 0.00965      (0.02  / 2.0717)
wall_damage  0.03  → 0.01448      (0.03  / 2.0717)
```
⚠ **"Neutral" means neutral AT THE REFERENCE TEMPERATURE, not
byte-identical everywhere** — that is the whole point of the patch:
below the reference fires burn SLOWER than today, above it FASTER. Two
predicted side effects to MEASURE, not assume:
- **G2 ramp may lengthen** (the ramp phase runs cooler than the reference
  → hotf < f_ref → less deposit early). 3a measured 80.3 s; if it leaves
  the 30–120 s window, that is a Phase-4 dial matter, not a design fault.
- Burnout time at reference ≈ unchanged; hotter fires burn out faster
  (G3's direction — with hotf_cap=10 a saturated fire destroys ~4.8×
  faster than today).

**Verification bench (the gate)**: rerun `_phase3a_driver.py --b1` both
legs + the M1 reference. Expect (a) open-control plateau ≈ preserved
(the re-size's job — fourth-root fixed point is self-consistent since
deposit at T_ref is unchanged); (b) **sealed infinite-fuel X_death moves
DOWN from 0.166 toward 0.13** — the feedback now fights the cold-collapse
by drawing harder as it heats, which is exactly the mechanism that should
finally make the O2 wall the proximate death cause; (c) report M1 ramp +
burnout deltas for the G2/G3 record.

## Session protocol (Erik, 2026-09-01): CPU-ONLY until session close

GPU is running Erik's civulator RL training — no CUDA builds, no
test_cuda_* runs during 3c. Every patch edits BOTH mirrors (code may not
drift) but verifies CPU-only. **Owed at session close, ONE batch: rebuild
`cpp/build_cuda`, run all test_cuda_* incl. `test_cuda_p68_fire.py`
against the final dials, fix any lockstep drift.** (R1's CUDA twin is
edited but unverified against the final 0.95 dial.)

## Ruling R2 (Erik, 2026-09-01): both loss channels stay — LOCKED

Physics frame: cool_shift is LINEAR in T (Newton-cooling/convection
proxy); radiation is T⁴. Both have real physical counterparts; keep both
always-on (T vs T⁴ does the regime-switching smoothly, no threshold).
cool_shift's SIZE (not existence) becomes a Phase-4 dial informed by the
re-derivation below. Radiation-only remains a bench option, not a ruling.

## The T* re-derivation (brief item 6 — done 2026-09-01, Fable)

**The corrected law.** For an emitting tile (burning, or solid ≥
T_emit_gate), per tick, from the code's actual updates
(raycaster.h rules 1–4, combustion.cpp H_bed deposit, temperature_solver
Pass 3):

```
deposit:   ΔT₊ = H_bed·O2_claimed / 2^his
rad loss:  ΔT₋ᵣ = a_s·Σ_rays w·(E°[T] − E°[T_partner]) / 2^his
                  (lone open-field emitter: all 8 rays → sky, partner=0;
                   E°[T] = rad_scale·K(T)⁴, K = 293+T post-G12;
                   flux limiter (T·2^his)>>4 per pair — NOT binding at
                   plateau (1.02e6 ≪ 1.51e7) nor at G1 temps (checked to
                   1300 K))
cool loss: ΔT₋c = T / 2^cool_shift
```

Equilibrium (deposit = losses), radiation dominating:

  **K(T*) = ( H_bed·O2_rate / (rad_scale·a_s·2^his·…) + K_amb⁴ )^(1/4)**

— a FOURTH-ROOT law: the plateau is a Stefan–Boltzmann equilibrium.

**Validation 1 — the shares at the measured plateau** (pre-R1 open
control: I=0.751, T=460.5 game, X=0.189; furniture a_s=0.5, his=3,
cool_shift=13, rad_scale=5.1427e-5): E°(460)−E°(0) = 1.632e7 counts →
rad loss ≈ **15.6 game/tick**; cool_shift loss = 460.5/8192 ≈ **0.056
game/tick**. **Radiation carries ≈99.6% of an emitter's losses;
cool_shift ≈0.4%.** Equal-loss crossover ≈ T=10 game — cool_shift is
negligible for EVERY emitter and is de facto the non-emitter/below-gate
channel already (Erik's two-regime intuition, realized structurally).

**Validation 2 — M5 predicted the fourth root.** 16× k_grow → I ×8, o2f
sagged → deposit ×≈5 → law predicts K ratio 5^0.25 ≈ 1.50; measured
814/568 = 1.43. The old linear-loss formula's 105–110× overshoot is
exactly the missing T⁴ term.

**Consequences (Phase-4 guidance):**
- G1 (753 → 1300 K) needs deposit ×≈9 ((1593/753)⁴… strictly
  (K₁⁴−K_amb⁴)/(K₀⁴−K_amb⁴) ≈ 8.8): **H_bed is the lever** (rad_scale
  also works arithmetically but double-books into ignition margins —
  avoid). k_grow/cool_shift are confirmed near-irrelevant for emitter T.
- The config.toml gain/T* comment (~:855 block) must be replaced with
  this law (patch rider on the R1 or next code patch).
- Status: float-estimate derivation with two independent empirical
  closures (plateau share + M5 fourth-root). Exact-integer replication
  (a script reading live planes for one tick) queued as a Phase-4
  instrumentation nicety, not a blocker.
