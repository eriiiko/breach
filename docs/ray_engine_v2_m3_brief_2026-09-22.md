# M3 brief — derived `H_bed`, the flip becomes mergeable (2026-09-22, amended 2026-09-23)

> **Status**: brief, written by Fable at Erik's request after the trajectory
> assessment of 2026-09-22. Narrowed on purpose. Not yet built.
>
> **Amended 2026-09-23** (Erik blessed the plan; the orchestrator applied it):
>
> - **The timer change is deferred.** The original §2 changes 2 and 3
>   (`wall_damage` → 0, the destroy decision keyed on the fuel store) and the
>   two properties that guarded them move to a fire arc after P7, under Erik's
>   rule of 2026-09-23: *one system at a time* (`autonomous-patch-workflow`).
>   His q5 rulings stand; only their timing moved.
> - **Erik plays the flip BEFORE it merges into `fire-12`.** That is the feel
>   iron rule, and thermal v2's T5 row says "built, gated, pushed, NOT merged".
>   The original brief had merge-then-play.
>
> **Purpose of the patch**: make `12-t5b-the-flip` green so the whole stack can
> merge into `fire-12`. Nothing else. Fire behaviour (ignition, spread, burn
> durations, how fires end), material design, tools, skills and CLAUDE.md rules
> are OUT (see §6).
>
> **Depends on**: `docs/ray_engine_v2_scheme_study_2026-09-13/report_t5b.md`
> §6.3 (the 2^16 measurement), `report_m2.md` §10 (what M3 inherits),
> `report_m2b_probe_output.txt` (the Fleck finding is an artifact; M3 is
> unblocked).

---

## 1. Where the stack stands

- `fire-12` is green at `cc67531` (main's pyright/clangd tooling merged in on
  2026-09-23). Nothing from this arc has reached it since T5a.
- `12-t5b-the-flip` carries the flip + M1 + M2 + the M2b probe (worktree
  `breach-t5b`).
- Suite at the flip tip: **3408 passed, 11 failed** (report_m2 §3).

**Pre-steps: DONE 2026-09-23 by the orchestrator.** M2b merged into the flip
(`b85c32c`), its branch and the `breach-fleck` worktree gone; this brief
committed on the flip; `12-m3-derived-hbed` cut off the flip in the worktree
`breach-m3`. One writer.

## 2. What M3 changes (two things)

| # | change | where |
|---|---|---|
| 1 | `H_bed` takes its derived value: solver-level `H_BED_M · 2^H_BED_SHIFT = 38.73`, i.e. mantissa 38.73, shift 0 | `config.toml:1214` / `:1227` (`[physics.combustion]` block), bound at `physics_runner.py:753` |
| 2 | The integer reference's default E-table becomes the sweep's derived scale (`rad_scale_derived = 1.6533e-08`, not the retired `[physics.fire] rad_scale = 5.1427e-5`), and `config_dials_match()` compares against the SWEEP's key, so it can see the drift it exists to catch | `sweep_ref_q.py:70` `RAD_SCALE` + `sweep_ref_q_gates.py`; **reference first, gates re-run, then nothing in the engine needs to move.** CLAUDE.md calls the reference THE executable spec of the live sweep, so it bakes at the live scale. M2b §4 measured the blast radius: gate 0 and the harness family are unaffected (they test parity on whatever table they bake), G12 flips green, nothing under `src/` or `cpp/` imports it |

**Why change 1 is a bug fix, not a model choice.** The shipped
`H_BED_M · 2^H_BED_SHIFT = 19 827 · 128` is 38.73 × 2¹⁶ to the last digit: T3
derived the constant per unit of N_O₂, and the engine multiplies a raw Q16.16
burn count (report_t5b §6.3).

**Trap on change 1.** `physics_runner.py:753` multiplies the config mantissa by
`self._o2_potency` before it reaches the solver. T5b's 38.73 was measured by
driving `CombustionSolver::step` directly, so 38.73 is the SOLVER value. If
`_o2_potency != 1` the config mantissa is `38.73 / _o2_potency`. Check, state
which, and pin the solver-level product in a test.

**The timer stays live.** `[physics.fire] wall_damage` keeps its shipped value
and the burn-through loop keeps its `wall_hp[i] <= 0` test, so fires end, and
tiles burn through, by the same mechanism as today. Erik's q5 rulings (timer →
0, destroy by chemistry, the 3-minute ruling retired) are deferred to the
post-P7 fire arc, not rejected. **T6 must therefore NOT delete the depletion
code: it is live.**

## 3. The eleven reds, and what each may legitimately do

Run the suite first and put the exact list in the report skeleton before
touching anything. From report_m2 §3 and §10:

| group | tests | expected under M3 | if still red |
|---|---|---|---|
| runaway guards | `test_eos_p4_combustion` ×3, `test_ps1_smoke_roundtrip`, `test_fire_feedback::test_conservative_default_does_not_firestorm_wood_room`, `test_eos_p4_combustion::test_e2e_2` | **clear on change 1 alone** (M2 §10 measured 5 of these) | a finding; do not bend |
| unit-state digests | `test_unit_state_digest` ×2, `test_s3c_unit_state_digest` ×1 | flip to "the fire never reached the unit" (M2 §10) | these are GOLDENS of a scenario. If the scenario no longer delivers fire to a unit the test is vacuous. Adjust the scenario so a fire reaches a unit (that is the property), then re-baseline ONCE with rationale naming Erik's q7 ruling. This is the iron rule's "once per approved change", not a bend |
| G12 | `gate12_damped_source_is_monotone` | an ARTIFACT of the reference baking at the retired scale; clears on change 2 (M2b: 0 backward steps on every row at the derived scale). So there are 10 real reds, not 11 | do not widen `F_SHIFT`; that is a P-series change |
| hot rail | `test_e1_hot_rail::test_no_rail_hits` | M2 measured 27 ceiling ticks against 14 even under derived `H_bed` | the load-bearing half (`t_max_phys_hits <= 8`) must hold. The 14 is a "2x measured headroom" budget calibrated on the 154 kg crate; re-state it from the measured value on the thin row, with the measurement in the test's docstring. If the load-bearing half fails, stop and report |
| payoff orderings | `test_payoff_orderings_perturbation_robust` | does not clear on `H_bed` alone (M2 §10) | diagnose and report what it is measuring now. Do not bend. If it needs a ruling, it goes in the §7 list and does not block the patch |

## 4. The one property to pin: a corrected fire sustains

Write this test first, run it on the tip, then make the change.

**A fire on a thin row SUSTAINS after ignition.** Bench:
`tools/fire_timing_harness.py --mat kindling`. Property: after the ignition
transient the tile keeps burning long enough to consume a stated fraction of
its fuel (say more than a quarter of the `wall_hp` bar) before intensity reaches
0. Assert the fraction loosely, never an exact duration.

**The knife edge, so nobody is surprised (M2b §5.2).** The engine's radiative
plateau for a tile fed a constant power reproduces physics to 1 %. At the
corrected fuel-bed share of 17.7 kW a thin tile plateaus at 285 game. Kindling,
foliage and furniture ignite at 280; wood at 300. So the corrected fire sits
within 20 % of a single number on both sides of sustain. Part of that margin is
deliberate: `H_fuel` stays 29x under its derived value until P5, so the flame
gas heats the panel far less than it physically would. **If the property fails
on every thin row, M3 still ships with the reds green and this property left
red BY NAME, and the choice is Erik's**: wait for P5's gas channel, spend #68's
skin node, or question the 25 % fuel-bed share. M3 does not pick.

**Post-M3 one-liner (M2b §5.3):** `m2b_fleck_live_probe.py` over a full burn
must report `min f == 1.0` on every thermal solid, now that nothing rails at
`T_MAX_PHYS`. Pin it as a test; it turns "the damping is inert" into a check.

Plus the standing gates: the arc #54 closure identity closes in int64
(`test_e1_hot_rail::test_no_transport_mint`, `test_gas_energy_tile_flip`,
`test_thermostat_books`); CUDA lockstep at tolerance 0 (`pytest tests -q -k
cuda`, both builds rebuilt); GOLDEN_AGGREGATE moves only if a digest field
changed, which none does here, so it should NOT move; if it does, that is a
finding.

## 5. Burn duration: one line, no steering

Run `--mat kindling`, `--mat wood`, `--mat furniture` once each and put three
numbers in the report: time to ignition, time to burn-out, peak tile
temperature. The timer still ends fires in M3, so these are the timer's
numbers, recorded as the baseline for the post-P7 fire arc. **Do not tune
anything toward any duration.** Erik's ruling of 2026-09-22: a fire that takes
hours to burn out is acceptable at this stage.

## 6. Out of scope, explicitly

- **The timer change** — `wall_damage` → 0 and destroy-by-chemistry, with its
  two properties ("a fire goes out", "a burnt-out tile is destroyed"). Deferred
  to the post-P7 fire arc (amendment above). The design for it is
  `docs/thin_material_rows_design_2026-09-20.md` §8.
- `H_fuel` stays at 4.0. The thin-rows design table lists 116.2 for M3, but
  T5b §5.3 showed the derived value is unshippable until the gas has its
  radiative loss channel, and the arc plan gives that to P5. Setting it now
  would rail the room and create a new compensating pair.
- No M4: no `derive_material_row.py`, no skill, no CLAUDE.md rules. **M4 is
  cut** (Erik, 2026-09-23).
- No material row changes. No new rulings. No thickness, mass or lumped-door
  edits. The material table is placeholders (Erik, 2026-09-22).
- No `F_SHIFT` widening (P-series).
- No re-fitting of any dial. If a test can only go green by fitting, it stays
  red and goes in §7.
- Not M3, recorded here so they are not lost (from M2b §6):
  - `bindings.cpp:2093`, `:296`, `:3537` take `unchecked<3>()` arrays with no
    shape check; a wrong-shaped `face_shift` reads out of bounds and gives a
    different answer per launch. Live callers are correct. **A standalone
    10-line patch straight on the flip branch, any time**, not inside M3.
  - CLAUDE.md's radiation-sweep row quotes "1068 game for wood" from three
    superseded premises (emissivity 1.0, thermal_mass 8, fitted scale). **T7.**
  - `physics_runner.py:1045` still says the old cast feeds `rad_net`. **T6.**

## 7. Report: at most 150 lines

`docs/ray_engine_v2_scheme_study_2026-09-13/report_m3.md`, skeleton committed
before any code. Sections: what landed (the two changes, with the
`_o2_potency` resolution), the red list before and after, the pinned property
and how it was validated by breaking it (ONE break is enough), the three burn
numbers, and a **"for Erik" list of at most five lines**. A finding in another
system is one line, never a fix (`autonomous-patch-workflow`, *one system at a
time*). A finding that would block the patch must be reproduced on the LIVE path
(derived table, real level) before it is called blocking; M2's §5.1 was not,
and it cost a probe session.

## 8. Execution

- Branch `12-m3-derived-hbed` off `12-t5b-the-flip`, own worktree
  (`breach-m3`), one writer.
- Model: **Opus**. Small brief, no resume beyond one.
- Builds: `cpp\build_cpu_home.bat` and `cpp\build_cuda.bat`, both, always.
  Python `C:/Users/steen/anaconda3/python.exe`. Tests `pytest tests -q`.
- Commits `feat(#12): M3 -- ...` / `test(#12): M3 -- ...` / `docs(#12): M3 -- ...`,
  explicit paths, never `git add -A`. Do not merge, do not push.
- Done means: suite 0 failed on both builds (or the §4 property red BY NAME,
  per §4), report under 150 lines. Then the orchestrator merges M3 into the
  flip with `--no-ff` and pushes. **Erik plays the flip branch** (the P3 feel
  gate the design always had) **before** the flip merges into `fire-12` with
  `--no-ff`.
