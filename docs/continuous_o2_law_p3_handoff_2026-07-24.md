# Continuous O₂→combustion law — P3 build handoff (2026-07-24, Opus)

Build of `docs/continuous_o2_law_design_2026-07-24.md` on branch
**`o2-continuous-law`**. This is the append-only capture doc for the joint
re-tune session (design §5). **This branch is PARKED — do NOT merge to main
until the re-tune** (Erik's call, 2026-07-24).

## What shipped

- **P1 + P1b** (CPU) — commit `547fb12`. Fire O₂ sustain factor is now LINEAR in
  the local O₂ **mole fraction** `X = Σn_o2/Σn_total` over open neighbours, with
  an extinction limit: `o2f = clamp01((X − X_ext)/(X_amb − X_ext))`. Replaces the
  smoothstep on absolute `n_o2`. `burn_rate 1.0 → 0.02`; combustion demand is now
  `burn_rate·I·o2f·dt` per claimant. Ignition reads the same law (`X > X_ext`).
  P1b: ignition re-seed requires `wall_hp > 0` (burnt-out tiles stay out).
- **P2** (CUDA lockstep) — cuda_fire.cu / cuda_combustion.cu mirrors, **GREEN**:
  GPU == CPU bit-identical (tol 0) — `test_cuda_p68_fire.py` +
  `test_cuda_p69_combustion.py` pass (fuzz + deterministic forcers + full
  130/120-tick lockstep trajectories), full `-k cuda` slice (16) green. Built +
  gated on the Lenovo (VS2022 BuildTools + CUDA v12.9, `build_cuda_lenovo.bat`).
  Independently re-confirmed by the orchestrator. Note: the combustion fixtures
  never populated `fire` (harmless under the old `(void)fire` law); seeded to
  `I=1` — where `demand` reduces to the old uniform `burn_cap` bit-for-bit, so
  every hand-tuned split/contention constant in the old fixtures still holds.
- Papers: header citations in place (Peatross & Beyler 1997, Huggett 1980);
  `docs/papers/continuous_o2_law_citations.md` records them — **PDFs still to be
  archived** (flagged there).

## Gate a (GREEN, on branch): `tests/test_continuous_o2_law.py` (14)

Endpoints (X=X_amb→o2f=1, X≤X_ext→0), midpoint linearity, X_ext=0 degeneration
to X/X_amb, **density invariance / trap fix** (thin ambient-composition gas still
burns; bit-exact in the clamped region), vitiation starves, fully-enclosed reads
0, determinism, plus ignition (X>X_ext) and P1b (no-fuel no-reignite).

## Expected REDS on this branch (by design — NOT breakage)

These fail **because the law/tuning changed and the re-tune is deferred**. They
rebaseline at the re-tune. Do not "fix" them here.

| Test file | Why red | Kind |
|---|---|---|
| `test_fire_feedback.py` (10) | calls `FireSimulation.step` without the new `n_total` arg | signature (TypeError) — WIP-staged |
| `test_temperature_ignition.py` (9) | `_GMapStub` lacks `wall_hp` + `inert_n2` plane the new ignition needs | stub (AttributeError) — WIP-staged |
| `test_fire_o2_invariant.py` (1) | old absolute-O₂ ignition predicate | behavioral |
| `test_eos_p4_combustion.py` (≈5) | `burn_rate 0.02` + `demand∝I` + flameless-ember change | behavioral/tuned |
| `test_eos_p5_1_stoich.py` (4) | ember fuel-consumption relied on flameless combustion draw | behavioral/tuned |
| digest / golden gates | sim-path change (design gate c) — **ONE** rebase at re-tune only | golden |

`test_simulation.py` (full live path) **passes** — the wiring runs end-to-end.

## Behavioral shifts to surface at HUMAN-TEST (Erik's eyes)

1. **Flameless-ember change.** `demand ∝ I`, so a hot tile with fire `I=0`
   (yesterday's emergent "ember") now draws **no** O₂ and deposits **no**
   combustion heat — "a choked/cool fire consumes nothing" (design §2.3). This
   supersedes the v2.5 "ember glows awaiting oxygen" mechanic (same anti-zombie
   direction as P1b). Confirm the feel is what you want.
2. **Decompression vs vitiation.** Under the mole-fraction law, a room that loses
   density but keeps ambient composition (21/79) no longer extinguishes via
   `o2f` — only true **vitiation** (products displacing O₂) or venting to
   flagged **vacuum** neighbours (dropped from the sums → X→0) does. This is the
   intended trap fix, but it IS a change from the old absolute-density gate.
3. **Trap closed.** Hot thin gas at ambient composition burns (gate-a proves it
   at unit scale; eyeball the planetside crate-in-a-room quirk #1).

## The joint re-tune (design §5) — the to-do for that session

Sponge-safe bench (≥ 80×36), old fire-tuning chat, AFTER this + sky-exchange merge:
- **`burn_rate` FIXED at 0.02** ("never touched again without a real reason").
- Re-fit **`k_grow`/`k_die`** (peak ≈ 0.6 @ ~3 min; expect the ratio → ~1:1 since
  `avail` now saturates at F≈0.7 — I_eq math, tuning-plan §5.2).
- **`wall_damage`** (≈ 0.083 for ~7.5 min burnout at Ī≈0.8 — bench decides).
- **`X_ext`** (`o2_frac_ext`, feel: how visibly fires gasp).
- Sky **`τ`** (from the sky-exchange doc).
- Then: **update/rebaseline the branch-red tests** (add `n_total` to direct
  `FireSimulation.step` callers; give the ignition stub `wall_hp` + an
  `inert_n2` plane; refresh tuned combustion/E2E expectations), **ONE golden
  rebase** with written rationale, **HUMAN-TEST**, merge.

## Implementation notes for the re-tune (durable facts)

- `n_total` denominator = Σ **conservative** bulk planes (O₂+N₂), soot excluded —
  the SAME real N_total the temperature Pass-1 deposit uses. Built once in
  `PhysicsEngine::step_tail` (before the fire step) and shared. Both the normal
  and GPU-resident ticks funnel through this one `step_tail`.
- Mole-fraction divide floor `X_N_FLOOR = quantize(0.01) = 655`: guards the
  per-cell `reciprocal_q16` and makes a near-vacuum non-vacuum cell read X≈0.
- `X_amb` is per-map — bound from the level's authored `[ambient] o2_frac`
  (0.21 fallback), refreshed each tick in `_ambient_args`.
- `o2_thresh_burn` retired as a gate; kept as an epsilon skip-floor. `P_min`/
  `P_full` + ignition `o2_threshold` tombstoned (still bound so old configs don't
  hard-error).
