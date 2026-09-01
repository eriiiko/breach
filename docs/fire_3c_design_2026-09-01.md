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
4. `I_cap_per_avail` re-sized 14.0 → **0.75** so the open-control ambient
   plateau stays where 3a/prebench measured it (I_eq ≈ c·a, a ≈ 1 at
   ambient now). One-time mechanical re-size; Phase 4 owns the taste pass
   (Erik's "hotter at lower I" = H_bed work, deliberately NOT this patch).
5. Config: `o2_frac_amb` becomes live (was fallback); `o2_frac_full = 1.0`
   retired with tombstone; `o2f_cap = 5.0` new; k_die comment's death-wall
   claim re-derived (now true).
6. Mechanically-implied golden re-baseline (sim behavior change), ONE,
   rationale = this ruling. Cross-mirror CUDA gates mandatory (S6 fire
   kernel twin).

**Verification gate (before Erik's feel pass)**: rerun prebench --b1 both
legs. Expect: open control plateau ≈ unchanged (I≈0.75, T≈460 game);
sealed infinite-fuel death moves DOWN from X=0.176 toward the foot
(exact X measured, T-gate may still be proximate — remaining gap feeds
the hot-burns-faster item). Report X_death, death cause, timeline vs the
2026-09-01 baseline.

## (pending) R2 — cool_shift vs radiation

Under discussion. Physics frame: cool_shift is LINEAR in T (Newton
cooling/convection proxy); radiation is T⁴ (gated by T_emit_gate for
non-burning solids, always-on for burning tiles). Both channels have real
physical counterparts (convection + radiation); question is their shares.
T* re-derivation (brief item 6) quantifies; radiation-only bench
(cool_shift → very high shift, runtime-patched) is the cheap experiment.
