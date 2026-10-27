# report_m3.md — M3: derived `H_bed`, the reference at the sweep's scale

> Brief: `docs/ray_engine_v2_m3_brief_2026-09-22.md` (amended 2026-09-23).
> Branch `12-m3-derived-hbed` off the flip at `596ccc7`, worktree `breach-m3`.
> No C++ touched; both binaries built once at the start. Not merged, not pushed.

## 1. What landed

**Change 1 — `H_bed` derived** (`ccc1da7`). `H_BED_M` 19827 → **38.73**,
`H_BED_SHIFT` 7 → **0**. Heat counts per RAW count of O₂ burned =
0.25 (Drysdale) × 13.1 MJ/kg (Huggett) × 0.36878 kg/65536 ÷ 0.4759 J = 38.73
(`V_tile` cancels). **`_o2_potency` resolved: it is 1.0 and folds into the
mantissa**, so the config value IS the solver value; measured on the live
`CombustionSolver.step`, 38.7274 vs the long-form 38.7443 (0.04 %).

**Change 2 — the reference guards the sweep's scale** (`ea2e3ef`).
`config_dials_match()` compares `RAD_SCALE_LIVE` with `[physics.radiation]
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

Tip `596ccc7`: **3408 passed, 11 failed, 4 skipped, 4 xfailed** (209 s).
M3: **3420 passed, 3 failed, 4 skipped, 5 xfailed** (260 s). CUDA `-k cuda`:
**25 passed**. `GOLDEN_AGGREGATE` unmoved (`e369b616…`); the #54 closure gates pass.

| # | test | at the tip | after M3 |
|---|---|---|---|
| 1 | `test_e1_hot_rail::test_no_rail_hits` | 32 ceiling ticks > 14 | green: soft budget 14 → 64 (2× measured 32, brief §3); load-bearing `t_max_phys_hits` = 0 ≤ 8 |
| 2 | `test_eos_p4_combustion::test_thermal_spike…` | peak 15 998 | green (change 1) |
| 3 | `…::test_e2e_1_sealed_room_fire_self_starves` | wall burned through | green (change 1) |
| 4 | `…::test_e2e_2_breach_vents_o2_and_kills_fire` | vented 0.96 > sealed 0.18 | **RED — finding 4**; vented 1.00, sealed 0.59 |
| 5 | `…::test_payoff_orderings_perturbation_robust` | `AttributeError: cool_shift` | **RED — finding 5**, untouched |
| 6 | `test_fire_feedback::…firestorm_wood_room` | 27 of 28 alight | green (change 1) |
| 7 | `test_ps1_smoke_roundtrip::…` | +2 counts at tick 4 | green (change 1) |
| 8 | `test_ray_engine_v2_integer_reference` G12 | 542/365/517 steps | green (change 2) |
| 9 | `test_s3c…::test_scenario_actually_kills_the_unit` | HP −6361 at tick 0 | green: HP 22 → 2.0 (q7; kill at tick 17) |
| 10 | `test_unit_state_digest::test_unit_digest_is_nontrivial` | both dead at tick 0 | green: fixtures in kW/m² (T5b's currency, not q7) |
| 11 | `…::test_perturbed_kill_event_diverges` | shifted kill not seen | green (same re-anchor) |
| 12 | `test_radiation_sweep_shadow_wiring::test_the_flux_sensor_ceiling…` | green | **RED — NEW, finding 6**, untouched |

`test_s3c…::test_kill_event_lands_on_the_same_tick_run_twice` passed at the tip
only because the runaway killed on tick 0 in both runs; under change 1 nobody
died (None == None fails); the HP re-anchor restores it.

## 3. The pinned property, and how it was validated by breaking it

`tests/test_m3_derived_h_bed.py`, written first, run on the tip, then the changes.

- **Sustain (brief §4).** The brief's wording — "a quarter of the bar before
  I → 0" — **cannot fail for its reason**, measured: with `H_BED_M = 0` kindling
  still spends 92 % and furniture 70 % before the flame goes out (the bench seeds
  AT ignition, `hot` stays open to ignition − 200, the timer eats the bar while
  the seed cools). Pinned instead: **the share of the bar spent while the tile is
  alight AND at or above its own ignition temperature > 25 %** — M2b's "both sides
  of sustain". Derived `H_bed`: **kindling 46.5 %** (self-held 8.4–25.3 s,
  plateau 284.6), **furniture 49.8 %** (7.5–69.7 s, 288.9), **wood 0 %** (a
  strict xfail — finding 2). **The break:** `H_BED_M` halved in config →
  kindling 0.0 % and furniture 0.0 % self-held (red) while the brief's metric
  still read 80.5 % / 61.8 % spent; config restored from git.
- **`H_bed` pin.** Solver-measured heat per raw O₂ count within 0.5 % of the
  long-form derivation. Tip: red (bound 2.538e6 vs 38.74). M3: 38.727, green.
- **Fleck inert over a full burn (M2b §5.3).** `fire_tuning`, ignition as the
  M2b probe, run to extinction: tip red with min f = **0.031830** (M2b's number
  to the digit); M3 **min f = 1.0 over 9 063 ticks**, fire out at ~377 s, peak
  solid 291.8 game. Control: the fitted scale shows damping within two ticks.

## 4. The three burn numbers (the timer's — a baseline, nothing tuned)

`tools/fire_timing_harness.py --mat <row> --max-seconds 900`, once each. The
bench seeds at ignition (t = 0), so "ignition" is the flame establishing.

| row | flame at 90 % of peak I | burn-out | peak tile T |
|---|---|---|---|
| kindling | 5.7 s | fuel exhausted 75.0 s (flame decaying, I 0.04 at 78 s) | 284.6 game |
| wood | 6.4 s | flame out 137.5 s, **82 % of the bar left** | 299.7 = its seed; never rises |
| furniture | 6.3 s | flame out 339.2 s, 8 % left | 288.9 game |

## 5. Findings (one line each; another system's finding is never a fix)

1. **Change 2's blast radius was 115, not 0** (§1); the default-table move is Erik's.
2. **Wood panels do not sustain** under the derived `H_bed`: never past 300, out at 137.5 s with 18 % spent; convection (κ 0.12) and emissivity 0.90 each sink it alone (31 % / 37 % without, diagnostic only). M2b's knife edge, on one row.
3. **The brief's §4 wording measures the timer**, not the flame (§3); replaced by the self-held share — Erik may veto.
4. **`e2e_2`: the sustain gate is density-blind.** X over Σn_total with a 0.01 floor: a room vented to 0.013 atm in 40 ticks keeps its flame at I = 1.0 (X 0.2–0.27) while its demand burns almost nothing; "vented weaker at t = 400" relied on the old cast's 3110×-fast cooling. M3 cannot move it (0.96 at the tip). Fire-sustain system; red, not bent.
5. **Payoff orderings** still sets the `cool_shift` plane T5b step 7 deleted (its missed consumer); without that line it reads sealed 0.5935 / vented 1.0000 / flooded 0.0870 — finding 4 again — and is bit-identical under the 1e-5 perturbation.
6. **NEW red, the old cast's flux-sensor ceiling:** its scene reached the cap only because the 2^16 deposit drove a thin wood tile past ~2 740 game; now it cools from its 1263 seed. Its own message prescribes "find a hotter one". Its premise ("a cap on unit heat damage") is stale since the flip — no consumer reads `rad_flux` (units read `rad_flux_sweep`) — and T6 deletes that cast. Re-scene or retire: a ruling.
7. **`test_e1_hot_rail` never sees M3**: `storm_probe.PF1B` pins `H_bed` = 18125·2⁴ (7 488× derived); M2's "27 under derived `H_bed`" does not reproduce (32 at the tip and after).
8. **`test_unit_state_digest`'s red was T5b's currency** (`heat_flux_to_temp` 4701.2: "mild" φ 60 = 13.5 MW/m²), not `H_bed`; the brief grouped it with the fire scenarios.
9. **`mul_q16` floors each claimant's deposit** at shift 0: effective `H_bed` on the live bench 38.42 (kindling, −0.80 %) / 38.54 (furniture, −0.48 %) — ~1 game of kindling's 4.6-game margin.
10. `foliage` cannot be placed on the bench: its id 9 is the CSV SPACE code, so the loader makes it ambient.
11. `test_emissive_table::_dials()` bakes at `[physics.fire] rad_scale` and compares with the reference default — equal by history; T6's deletion of that key will break it.
12. `CombustionSolver`'s C++ defaults (`H_BED_M` 25290, shift 3) are a stale P-R4 value for direct-binding callers; config binds the live one.
13. Suite time grows ~50 s (209 → 260 s): the full-burn Fleck test (~36 s) and the three sustain runs.

## 6. For Erik

1. **Play the flip**: fires now plateau at ~285–292 game, not 16 000; kindling burns out its fuel in 75 s, furniture's flame lasts ~340 s.
2. **Wood panels do not self-sustain** under the derived `H_bed` (finding 2): P5's gas channel, #68's skin node, or the 25 % share — your choice; M3 does not pick.
3. **Three reds left by name**: `e2e_2` + payoff orderings (a vacuum-breached fire keeps full intensity — the sustain law reads mole fraction, not density) and the old cast's flux-sensor ceiling (re-scene it, or retire it with the cast at T6).
4. **The reference's default table** stays a resolving scale; moving it as the brief says costs re-plumbing every arithmetic gate (§1). Yes or no?
5. **The §4 criterion** is "fuel spent while self-held ≥ ignition", not "a quarter before out" — the latter cannot fail (§3). Veto if you disagree.

## 7. Orchestrator addendum (2026-09-23, after Erik's play test)

Verified independently: 3420 passed / 3 failed, the three named. Erik played
it on `playground` ("everything looks great"). Then: payoff orderings' dead
`gmap.cool_shift` write removed (T5b's missed consumer); the three reds marked
`xfail(strict=True)` by name — the two vented-room tests on #7 (its own patch,
after T7, before P4), the old-cast flux ceiling on T6. Rulings: the reference
default stays a resolving scale (§1); the §4 self-held criterion accepted.
