# report_m3.md — M3: derived `H_bed`, the reference at the sweep's scale

> Brief: `docs/ray_engine_v2_m3_brief_2026-09-22.md` (amended 2026-09-23).
> Branch `12-m3-derived-hbed` off the flip at `596ccc7`, worktree `breach-m3`.
> Status: IN PROGRESS — skeleton committed before any code; filled as it lands.

## 1. What landed

**Change 1 — `H_bed` derived** (`ccc1da7`). `H_BED_M` 19827 → **38.73**,
`H_BED_SHIFT` 7 → **0**. Heat counts per RAW count of O₂ burned =
0.25 (Drysdale) × 13.1 MJ/kg (Huggett) × 0.36878 kg/65536 ÷ 0.4759 J = 38.73
(`V_tile` cancels). **`_o2_potency` resolved: it is 1.0 and folds into the
mantissa**, so the config value IS the solver value; measured on the live
`CombustionSolver.step`, 38.7274 vs the long-form 38.7443 (0.04 %).

**Change 2 — the reference guards the sweep's scale** (`ea2e3ef`).
`config_dials_match()` now compares `RAD_SCALE_LIVE` with `[physics.radiation]
rad_scale_derived`; `E_LIVE` is baked at it; G12 measures the shipped rows and
its Q16 control on `E_LIVE` (0 backward steps on every row; control 14) and
keeps its pathological-corner mechanism on the resolving table P0b built it on.
Oracle: full and fast gate output byte-identical to the tip's except the config
line, one table line and G12. **ALL GATES PASS.**

**Deviation from the brief, measured — Erik's call (§6).** The brief moves the
reference's *default* table to the live scale, "gate 0 and the harness family
unaffected". Done literally, **115 tests go red**: gate 0 ×96, G4/G5/G10/G12
and their C++ twins, the frozen scalar-era digest, ambient-plane item 3, M1's
exponent test, the i64 table-top chain, `test_emissive_table` — every one a
non-vacuity pair (no f < 2²⁴, a clamp that never binds, a 9600-tick equilibrium
that does not converge). **None is an arithmetic disagreement**: C++ equals the
reference bit for bit at the live scale. The default therefore stays a
*resolving* scale, now documented as that and read from nowhere; moving it means
re-plumbing every arithmetic gate to name its scale — a third change.

## 2. The red list, before and after

Baseline at the tip `596ccc7`, both binaries rebuilt, `pytest tests -q`:
**3408 passed, 11 failed, 4 skipped, 4 xfailed** (209 s) — report_m2 §3 exactly.

| # | test | failure at the tip |
|---|---|---|
| 1 | `test_e1_hot_rail::test_no_rail_hits` | 32 ceiling ticks vs budget 14 |
| 2 | `test_eos_p4_combustion::test_thermal_spike_is_pre_existing_not_a_p4_regression` | peak 15 998 game vs 9 000 |
| 3 | `test_eos_p4_combustion::test_e2e_1_sealed_room_fire_self_starves` | wall burned through, 10.9 % hp left (want > 20 %) |
| 4 | `test_eos_p4_combustion::test_e2e_2_breach_vents_o2_and_kills_fire` | vented I 0.9585 vs sealed 0.1795 |
| 5 | `test_eos_p4_combustion::test_payoff_orderings_perturbation_robust` | `AttributeError: 'GameMap' object has no attribute 'cool_shift'` |
| 6 | `test_fire_feedback::test_conservative_default_does_not_firestorm_wood_room` | 27 of 28 wood tiles alight |
| 7 | `test_ps1_smoke_roundtrip::test_ps1_roundtrip_all_gas_planes_never_exceed_start` | +2 counts of gas mass at tick 4 |
| 8 | `test_ray_engine_v2_integer_reference::test_damped_source_is_monotone_in_temperature` (G12) | 542 / 365 / 517 backward steps on the thin rows |
| 9 | `test_s3c_unit_state_digest::test_scenario_actually_kills_the_unit` | HP −6361 at tick 0 and at the end (never "dropped") |
| 10 | `test_unit_state_digest::test_unit_digest_is_nontrivial` | 2 distinct digests in 36 ticks |
| 11 | `test_unit_state_digest::test_perturbed_kill_event_diverges` | shifted kill not seen |

After M3: _pending_.

## 3. The pinned property, and how it was validated by breaking it

_pending_

## 4. The three burn numbers (the timer's, a baseline — nothing tuned)

_pending_

## 5. Findings (one line each; another system's finding is never a fix)

_pending_

## 6. For Erik

_pending_
