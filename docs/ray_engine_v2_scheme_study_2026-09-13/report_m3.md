# report_m3.md — M3: derived `H_bed`, the reference at the sweep's scale

> Brief: `docs/ray_engine_v2_m3_brief_2026-09-22.md` (amended 2026-09-23).
> Branch `12-m3-derived-hbed` off the flip at `596ccc7`, worktree `breach-m3`.
> Status: IN PROGRESS — skeleton committed before any code; filled as it lands.

## 1. What landed

_pending_ — change 1 (derived `H_bed`, and how `_o2_potency` resolved);
change 2 (the integer reference bakes at `rad_scale_derived`).

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
