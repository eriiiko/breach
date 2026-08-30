# Fire/combustion test-family triage — 2026-08-30

22 pre-existing fire/combustion-family test failures (branch `gas-energy-arc`
at `45050f3`), triaged and dispositioned on branch `fire-tests-triage`. Per
Erik's delegated rule: restate a test to the property it protected; delete
only when its subject no longer exists, named in the commit.

Two items were reserved for Erik up front and left untouched (see
"Reserved for Erik" below); triage surfaced a third mid-fix (the P5.1 ember
lifecycle), also left for Erik rather than resolved unilaterally.

## Triage table

| Test | Property protected | Failure mode | Root-cause commit | Class | Action |
|---|---|---|---|---|---|
| `test_eos_p4_combustion.py::test_combustion_pass_conserves_o2_n2_soot_exactly` | Every combustion transaction moves mass O2->(smoke,N2) with zero net change to the O2+N2+soot sum | Never burned — `fire[]` was never seeded, and demand is `∝ fire[i]` since 547fb12 | 547fb12 (2026-07-24, continuous-O2 law) | Mechanical | Seeded `fire[cy,cx]` nonzero |
| `test_eos_p5_1_stoich.py::test_fuel_decrement_exact_and_deterministic` | The fuel decrement is the exact integer transaction the shipped solver computes | Never burned (same cause); pinned expected value also used the pre-547fb12 formula (no o2f/fire factor) | 547fb12 | Mechanical | Seeded `fire[]`; re-derived expected via a new `_read_o2f_exact` probe (reads o2f_j from the real solver rather than porting `reciprocal_q16`'s Newton iteration into Python) |
| `test_eos_p5_1_stoich.py::test_one_lsb_floor_never_crossed` | The 1-LSB fuel floor is never crossed, all 4 neighbours still draw O2 in the same tick | Never burned | 547fb12 | Mechanical | Seeded `fire[]` (fix already sufficient — passed unmodified otherwise) |
| `test_eos_p5_1_stoich.py::test_no_destruction_originates_from_combustion` | Combustion chars a tile to the floor but never destroys it | Never burned | 547fb12 | Mechanical | Seeded `fire[]` (fix already sufficient) |
| `test_eos_p6_9_isotropy.py::test_isotropy_bit_exact_zero_remainder` | 4-fold-symmetric contested O2 split is bit-identical across all 4 arms | Never burned; separately, `burn_rate=1.0`'s demand never actually exceeded the centre O2 counts used (uncontested branch, not the contested one the test is named for) | 547fb12 | Mechanical | Seeded `fire[]` identically at all 4 arms; `_comb()`'s `burn_rate` 1.0->4.0 (measured: restores genuine contention) |
| `test_eos_p6_9_isotropy.py::test_isotropy_bounded_bias_nonzero_remainder` | Same isotropy claim, non-divisible remainder case | Same as above | 547fb12 | Mechanical | Same fix (shared `_plus_scene`/`_comb`) |
| `test_pr3_capacity_law.py::test_fire_T_ext_is_derived_from_ignition_temp` | `fire_T_ext[mat] = ignition_temp[mat] - ignition_to_ext_delta` | Pinned literals (180.0/200.0) were downstream of the OLD delta (100); shipped delta is now 200 | 9016cd7 (2026-08-13, P-K0 dial promotion) | Mechanical | Reads the shipped delta from config; derives both expected values from it instead of pinning literals |
| `test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood` | The full radiative chain (emitter->ray->pair->rad_net->Pass-1) can ignite an air-separated wood target | Marked `xfail(strict=True)`; its own docstring named an unexpected PASS as the handoff signal for P-F1b landing — it has | P-F1b recalibration (already landed) | Mechanical | Removed the `xfail` marker per its own instructions; normal assertion |
| `test_fire_o2_invariant.py::test_production_ignition_matches_cpp_gate_off_tie` | Python's production ignition O2 gate matches the reference gate at every swept value | `_Cross3x3` fixture was missing `ignition_armed`, `wall_hp`, and an `inert_n2` gas mapping the continuous-O2 mole-fraction gate now reads | 423cd38 (edge-trigger arm) + 547fb12 (P1b fuel gate + mole-fraction law) | Mechanical | Added all three; seeded N2 so O2+N2 pins a unit total (swept value == mole fraction) |
| `test_eos_p5_1_stoich.py::test_lifecycle_ember_reignite_charout` | Full ignite->starve->ember->reignite->char-out lifecycle | `apply_temperature_ignition(..., o2_threshold=...)` — kwarg renamed | 547fb12 | Mechanical (fix), but see "Flagged for Erik" below | Kwarg renamed; SEPARATE issue surfaced, marked `xfail(strict=False)` |
| `test_eos_p4_combustion.py::test_e2e_1_sealed_room_fire_self_starves` | A sealed-room fire self-starves as local O2 depletes, not fuel exhaustion | Fire never fully extinguishes within the 520-tick budget (k_die's e-fold is now ~3000 ticks) | 9016cd7 | Restate | Restated to a genuine, monotonic, substantial post-peak decline (measured 23.2% off peak by tick 2000, asserts >=15%) + a re-derived pressure threshold (2%, measured 2.52%). Post-flame ember-signature checks dropped (out of this horizon's reach); lifecycle stays gated in `test_eos_p5_1_stoich.py` |
| `test_eos_p4_combustion.py::test_e2e_2_breach_vents_o2_and_kills_fire` | Venting kills an established fire faster than sealing it | Neither arm dies within 400 ticks (measured: vented 0.41, sealed 0.70 at t=400, both alight) | 9016cd7 | Restate | Restated ticks-to-extinguish -> intensity-at-fixed-tick; same O2-differentiation claim, visible without extinction |
| `test_eos_p4_combustion.py::test_e2e_4_inert_flood_smothers_fire` | An inert-N2 flood smothers a fire faster than leaving it alone | Same as above (flooded 0.39, control 0.70 at t=400) | 9016cd7 | Restate | Same fix |
| `test_eos_p4_combustion.py::test_payoff_orderings_perturbation_robust` | The O2-differentiation payoff ordering + timing survive a 1e-5 dial perturbation (not a chaos artifact) | No arm extinguishes within 400 ticks even at the O2-axis `cool_shift=9` pin | 9016cd7 | Restate | Renamed `_payoff_timings`->`_payoff_intensities`; measured baseline vs perturbed are now bit-identical (0.0 relative diff) — independently confirms the module's own "THE SPIKE IS GONE" (P-R4) finding; kept a loose 5% window as a regression trip-wire, not a re-derivation |
| `test_fire_feedback.py::test_cold_fire_decays_to_zero` | A cold fire (hot=0) decays all the way to 0 | Stuck nonzero at 200 ticks | 9016cd7 | Restate | Measured: genuinely still reaches exactly 0, at tick 2334. `max_ticks` bumped with margin; property unchanged |
| `test_fire_feedback.py::test_low_o2_fire_decays_to_zero` | An O2-starved fire decays to 0 | Same | 9016cd7 | Restate | Measured DIES at tick 2334 (same ODE as cold case); bumped with margin |
| `test_fire_feedback.py::test_vented_room_extinguishes` | A vacuum-vented fire decays to 0 | Same | 9016cd7 | Restate | Measured DIES at tick 2400; bumped with margin |
| `test_fire_feedback.py::test_burnout_when_wall_hp_runs_out` | Fuel exhaustion starves and kills a fire | Stuck nonzero at 500 ticks | 9016cd7 | Restate | Measured DIES at tick 4920 (bare 3x3 `FireSimulation.step`, ~0.02s for 5000 ticks); bumped with margin |
| `test_fire_feedback.py::test_wind_blows_out_a_small_fire` | Wind blows out a small/marginal fire (crossover with `test_wind_fans_a_big_fire`) | Neither calm nor windy dies within 6000 ticks | 9016cd7 (`k_wind_strip` 0.5->0.0) | Restate | `k_wind_strip=0.0` switches the blow-out term off entirely (not slow — OFF). Measured: this scene isn't even "marginal" any more (`I_cap_per_avail` 2.53->14.0) — both calm and windy GROW to a real equilibrium, windy strictly ahead throughout (opposite of the old claim). Restated to document exactly that, plus an assertion that `k_wind_strip` stays 0 so a future change is caught. **Deviates from the triage brief's suggested wording** ("windy and calm decay identically") — measured behavior is substantial divergence (fanning), not identity; the restatement documents what was actually measured |
| `test_s3b_fire_determinism.py::test_fire_field_and_burnthrough_list_bit_identical_run_twice` | Two runs of the same ignite->firestorm->starve->extinguish trajectory are bit-identical, and a real extinguish flip is exercised (non-vacuous) | 90 ticks no longer reaches the extinguish flip | 9016cd7 | Restate | Measured: still fully extinguishes, at tick 5571 (~2.6s/run). `TICKS` 90->6200 — cheap enough to just extend rather than drop the non-vacuity requirement; the bit-identity property itself was never at risk |
| `test_fire_feedback.py::test_plume_raises_own_atmosphere_wind_points_outward` | A burning tile's own heat raises its temperature and pushes wind outward | Isolated a plume->T shim deleted at 25a9823 (2026-07-31) — nothing left to isolate | 25a9823 | Rewrite | Checked `tests/_fire_bench.py` and the P4 e2e tests for redundant coverage first (neither covers wind direction) — genuine gap. Rewritten as a pipeline-level `PhysicsRunner` tick loop on the P4 sealed-room fixture: asserts the fire's own T rises over 60 ticks, and (with non-limiting O2 refilled each tick, removing the real O2-suction transient the rewrite surfaced) wind at its 4 neighbours nets outward at every tick from 35 on |

Two more failures in the baseline run are **out of scope**, unrelated to
fire/combustion, and were left exactly as found:
`test_cool_shift_axis.py::{test_every_material_carries_the_column_seeded_at_the_old_global, test_a_crate_grid_from_config_is_uniform_today_but_addressable}`
(wood `cool_shift` 13 vs 5).

## Reserved for Erik (untouched by this triage, per the brief)

1. **`test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing`** —
   a radiation-range question (320 behind a directional weapon). Left
   failing, untouched.
2. **`k_wind_strip == 0.0` (config.toml, P-K0 9016cd7)** — the wind blow-out
   mechanism (`k_wind_strip*W*(1-I)*I`) is dormant by shipped config, not by
   a bug. `test_wind_blows_out_a_small_fire` now documents this and asserts
   the dial stays 0 rather than silently re-passing a stale claim. Question
   for Erik: is the blow-out crossover intentionally retired (part of the
   "plume self-blow-out off, 2026-07-23" note), or should this dial have
   moved with the rest of the 9016cd7 promotion?

## Flagged mid-triage — a third item for Erik

**`test_eos_p5_1_stoich.py::test_lifecycle_ember_reignite_charout`** — the
brief's assigned fix (kwarg rename `o2_threshold`->`o2_frac_ext`, 547fb12)
applied cleanly, but uncovered a SEPARATE, deeper issue once the `TypeError`
stopped masking it: measured directly, the wood tile's own temperature
settles at ~15.5 game within ~6 ticks of ignition (driven only by the H_bed
fuel-bed deposit now that the "painter" is retired, P-R4) and never
approaches `IGN_WOOD_Q16` (300) for as long as the flame burns. Phase B's
scripted ember state needs `T >= ignition_temp` at the exact moment the
flame reaches zero (`combustion.cpp`'s own claim gate for a non-alight
source, `if (!alight && Tsnap[i] < ign_i) skip`, line ~508) — so the ember
this test builds appears **structurally unreachable via natural burnout**
under current physics, independent of any dial value.

This is a design-level question, not a test-wording fix: does decisions
#17's ember mechanic (P5.1) still fire in the shipped game at all, or does
the non-alight claim gate need its own lower "sustain" threshold — the same
kind of hysteresis fix P-R4 already gave the alight/ignite path (a tile that
IS alight burns on below its own ignition temperature; a tile that is NOT
yet alight, or has gone fully cold, currently has no equivalent floor)?
Marked `xfail(strict=False)` with the full measurement in its own docstring
rather than reinterpreted unilaterally. A `_starve_to_zero` helper (temporary
`k_die` crank to 2.0, its own pre-P-K0 value, mirroring Phase D's existing
`fuel_per_o2` crank) is included so the test's SEPARATE, already-diagnosed
9016cd7 slow-decay issue does not also block it once the ember question is
resolved.

## Final inventory

Full suite, `pytest tests -q` from the worktree root: **2313 passed, 30
skipped, 4 xfailed** (3 pre-existing + the 1 new ember flag above), **3
failed** — exactly the 2 `test_cool_shift_axis.py` failures (unrelated,
left alone) and `test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing`
(Erik's call, untouched). Matches the target inventory exactly.
