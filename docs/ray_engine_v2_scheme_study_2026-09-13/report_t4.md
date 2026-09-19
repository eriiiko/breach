# T4 — retire the dead test surface, before it can go red (issue #12)

**Branch** `12-t4-test-retirement` off `fire-12` (`29f7f49`, T3b merged).
**Worktree** `C:\Users\steen\projects\breach-t4`. Written incrementally; D1
lands before a single deletion.

Why this patch runs *before* the flip: the flip deletes `cool_shift` and
re-points the fold at the sweep, and at that instant every test pinning the old
heat law or the ambient decay is wrong. The original plan scheduled these
deletions *after* the flip, so the flip's gate would have been a red suite
(L3 critique B3, `docs/thermal_model_v1_critique_L3_scope_2026-09-19.md` §3).
Retiring the surface now is safe because the replacement gates were written at
P1/T1 against the shadow sweep and are green today.

**The rule this report is written to**: every deletion names the property that
replaces it and **where that replacement lives**. Where I cannot point at one,
the test is a finding and it stays (§5).

**Baseline**, this worktree, fresh CPU build: `2631 passed, 29 skipped,
4 xfailed` in 127 s, 0 failed.

---

## 1. D1 — the inventory

Legend for **replacement**: a `tests/...::name` is a test that exists and is
green **today**; "REMOVED by R*" means the property is deleted on purpose by a
ruling, and the channel that takes over its job is named.

### 1.1 Group A — the old radiation law (design v3 §11.1; 46 tests, four files)

`docs/ray_engine_v2_design_v3_2026-09-15.md` §11.1 is the instruction and its
per-file dispositions are followed exactly. All four files drive
`bp.Raycaster()` — the cast the flip deletes — so each module dies with its
subject.

#### A1 · `tests/test_pf1a_radiation_books.py` — 11 collected — **delete all**

Subject: the P-F1a "verified books" of the old cast — rule-1 pairs, rule-2 half
weight, rule-3 contact inertness, rule-4 sky by emitter, the `T_emit_gate` knife
edge, `rad_net.sum() + rad_amb.sum() == 0`.

| test | what it pins | replacement |
|---|---|---|
| `test_gate_i_equivalence_open_field_vs_sealed_ambient_room_tol_zero` | one emitter's net loss identical open-field vs centred in a sealed ambient room | `tests/test_radiation_sweep_gates.py::test_g2a_uniform_ambient_is_a_per_cell_exact_fixed_point_with_bodies_and_leak` |
| `test_gate_i_equivalence_a_half_walls_with_derived_tolerance` | the same at `a = 0.5`, tolerance derived from the contact-termination count | the same G2a — now exact, with no tolerance, because contact termination is gone |
| `test_gate_ii_ledger_identity_pre_fold_on_a_firestorm` | `Σ rad_net + Σ rad_amb == 0` pre-fold | `tests/test_radiation_sweep_gates.py::test_g1_three_term_identity_is_exact_with_each_sum_nonzero` (three terms) + `tests/test_radiation_sweep_ambient_plane.py::test_item4_conservation_is_exact_with_a_per_cell_ambient_and_the_leak_live` |
| `test_gate_iii_equal_temperature_lattice_nets_exactly_zero` | an isothermal emitter lattice exchanges nothing | `tests/test_radiation_sweep_gates.py::test_g2b_enclosed_isothermal_box_inner_layer_is_exactly_zero_with_f_below_one` — §11.1 names this line explicitly |
| `test_gate_iv_two_temperature_box_exchange_is_one_times_the_pair_law` | the rule-2 **pair law** at half weight, averaged over the 8-tick fan period | REMOVED by design v3 §2.3: a gather sweep has no pairs, no half weight, no fan. The per-face exactness that replaces it is gate 0, `tests/test_radiation_sweep_reference.py::test_cpp_sweep_reproduces_the_reference_bit_for_bit` |
| `test_gate_iv_crossing_the_emit_gate_is_continuous_both_directions` | continuity across `T_emit_gate` | REMOVED: `T_emit_gate` is deleted at the flip (design v3 §11 P3). Every cell emits, and the emission map is `E°(T)`, gated by `tests/test_emissive_table.py` |
| `test_gate_iv_boundary_tile_pinned_exactly_at_the_gate` | a tile pinned at `quantize(T_emit_gate)` | REMOVED with the gate |
| `test_gate_v_sealed_equal_T_room_wider_than_the_old_max_range` | no corridor leak past `max_range` | REMOVED: range dies with `RADIATION_RANGE` / `range_base` / `range_per_intensity`; the sweep traverses the whole grid. The zero it asserted is G2b |
| `test_gate_v_open_field_lone_emitter_matches_the_grey_body_rate` | the sky ledger equals the full grey-body rate | `tests/test_emissive_table.py::test_e_bucket_of_matches_the_reference_including_sub_ambient_and_saturation` (the rate) + G1's `rad_amb` term (the ledger) |
| `test_gate_vi_hot_sub_gate_solid_heats_a_cooler_emitter` | sign correctness below the gate | REMOVED with the gate; sign is structural in gather form, covered by G1's signed identity and gate 0 |
| `test_gate_vii_limiter_engaged_pair_approaches_monotonically` | monotone approach under the old flux limiter | `tests/test_radiation_sweep_gates.py::test_g5_radiative_substep_never_exceeds_max_t_before_e_inv_phi` and its four `test_g5_*` siblings — the maximum-principle clamp replaced the limiter |

Module-wide, "the Pass-1 rails are INERT in a gate run" is
`tests/test_radiation_sweep_gates.py::test_g4_marine_beside_ambient_wall_leaves_every_counter_silent_for_24_ticks`.

#### A2 · `tests/test_pr1_fire_plane_cast.py` — 11 collected — **delete all**

| test | what it pins | replacement |
|---|---|---|
| `test_equal_temperature_pair_nets_exactly_zero` | an isothermal pair nets zero | G2b |
| `test_self_cell_is_wholly_excluded_and_open_air_loses_to_sky` | self-cell exclusion; open air pays its sky | G1's three-term identity (the ring's two books) + `tests/test_radiation_sweep_ambient_plane.py::test_item3_a_hull_wall_facing_vacuum_radiates_against_a_different_ambient` |
| `test_exchange_conserves_exactly_and_flows_hot_to_cold` | exact conservation, hot to cold | G1 |
| `test_conservation_holds_over_a_600_emitter_firestorm` | conservation at scale | G1 (randomised scenes, both transports, S16/S12, leak on and off) |
| `test_air_neither_absorbs_nor_receives` | air's Kirchhoff inertness | `tests/test_optics_ingress.py` (the materials door: `heat_atten > 0` implies `thermal_mass > 0`; air is `a = 0`) + G2's mixed scenes |
| `test_touched_tile_set_matches_the_unchanged_fan` | the 8-ray fan's touched set | REMOVED: no fan. The traversal is pinned bit for bit by gate 0 |
| `test_emitter_enumeration_is_row_major_and_includes_warm_solids` | row-major source enumeration | REMOVED: gather form has no source enumeration; gate 0 pins the whole output |
| `test_emissive_table_is_the_exact_integer_bake` | the `E°` bake is the exact int64 chain | `tests/test_emissive_table.py::test_raycaster_and_engine_tables_are_identical_and_equal_the_reference_bake` — §11.1 names this move ("the bake test moves to `emissive_table.h`'s own test") |
| `test_air_gets_flux_but_no_energy` | flux without energy on air | G1's `rad_flux` term, asserted individually non-zero |
| `test_fan_rotation_connects_every_neighbour_within_ray_count_ticks` | D4 fan rotation coverage | REMOVED with the fan. Angular coverage is S16 half-offset: `tests/test_radiation_sweep_constants.py::test_half_offset_keeps_every_ordinate_off_the_axes` + `tests/test_radiation_sweep_gates.py::test_g6_isotropy_shear_rounder_than_step_at_one_tile_and_five_by_five` |
| `test_fan_rotation_is_a_pure_function_of_the_tick` | rotation determinism | REMOVED with the fan: the sweep carries no per-tick angular state |

#### A3 · `tests/test_fire_heat_source.py` — 14 collected — **delete 10, re-home 4**

§11.1: *"delete 10, rewrite 4"*, naming the four. The module's whole scaffold
(`_FireScene`, `cast_fire_heat`, `bp.Raycaster`) dies, so the file goes; the
four named properties are re-homed — two into gates that **already exist** and
carry them today, two into new pre-flip sweep tests (§2).

Delete outright (10):

| test | what it pins | replacement |
|---|---|---|
| `test_fire_deposits_heat_on_source_and_radiates` | the old cast's source deposit and falloff | emission is `E°(T)` (`tests/test_emissive_table.py`); the transport is gate 0 |
| `test_contact_faces_are_radiation_inert` | rule 3 | REMOVED by design v3 §2.3 |
| `test_no_fire_no_heat` | no emitters means no deposit | G2a — an all-ambient scene is an exact per-cell fixed point |
| `test_hotter_fire_reaches_farther` | `range_base + range_per_intensity` | REMOVED: range dies at the flip. The pre-flip reach property is re-homed, §2.1 |
| `test_wall_blocks_fire_heat_clear_path_heats_further` | occlusion by `heat_atten` | re-homed as the extinction-ordering gate, §2.3 |
| `test_heat_lands_on_solid_not_lost_in_air_conversion` | the deposit lands on solids, not air | `a = 0` on air at the materials door (`tests/test_optics_ingress.py`) + G2's mixed scenes |
| `test_full_chain_radiation_heats_air_separated_wood` | the old law warms a target across a gap | re-homed as the pre-flip reach property, §2.1 |
| `test_unit_away_from_fire_unharmed` | a distant unit takes exactly zero | `tests/test_unit_heat_damage.py::test_cold_tile_zero_damage` — mechanism-independent (it injects into `gmap.heat` directly) and survives the flip untouched |
| `test_cast_fire_heat_does_not_touch_rng` | `cast_fire_heat` draws no RNG | REMOVED: `cast_fire_heat` and both call sites are deleted at the flip. The sweep is pure integer with no RNG at all — gate 0, and `tests/test_no_float_in_sim_tu.py` pins `radiation_sweep.cpp` at 0/0/0 on the float ratchet |
| `test_lone_fire_does_not_firestorm_in_a_couple_ticks` | a `k_fire_heat`-era calibration guard | REMOVED: `k_fire_heat` is already dead (CLAUDE.md) and the flip recalibrates on `rad_scale_derived` (P2b). Its post-flip successor is verification item 10 of `docs/thermal_model_v1_design_2026-09-19.md` §6, owned by the flip's HUMAN-TEST |

Re-home (4):

| test | what it pins | where the property goes |
|---|---|---|
| `test_full_chain_heat_ignites_air_separated_wood` | heat crosses an air gap and reaches ignition | **new test**, §2.1 — the shadow sweep's fluence at a gap, read through `E°⁻¹(Φ)` against wood's own `ignition_temp` |
| `test_unit_next_to_fire_loses_hp_and_zombie_takes_4x` | a unit beside a fire burns; a zombie takes 4× | the 4× half **already lives** in `tests/test_unit_heat_damage.py::test_zombie_takes_fire_multiplier_more` and `tests/test_damage_pipeline.py:312` (`dmg *= CFG.zombie.fire_damage_multiplier`, pinned to the bit). The *radiation-reaches-a-body* half is a **new test**, §2.2 |
| `test_determinism_bit_identical_temperature` | same scene, bit-identical `temperature` after N ticks | `tests/test_w6_armory.py:583` — `trajectory_digest(traj) == GOLDEN_AGGREGATE`, the canonical whole-field determinism gate — plus gate 0's 34 bit-for-bit configurations |
| `test_fire_heat_is_wired_into_simulation_step` | the cast runs inside the tick | `tests/test_radiation_sweep_shadow_wiring.py::test_ab_default_scenario_is_wrap_free_and_the_sweep_runs_every_tick` and `::test_sweep_on_a_real_level_is_non_trivial_after_one_tick` — the sweep is step 2b, asserted on a real level |

#### A4 · `tests/test_heat_attenuation.py` — 10 collected — **delete 8, rewrite 2**

Subject: `heat_atten` as the 4th channel of the render march
(`cast_source_directional`). The march has no heat channel after the flip.

Delete (8):

| test | what it pins | replacement |
|---|---|---|
| `test_heat_passes_freely_through_air` | `heat_atten = 0` is bit-identical to no plane | the sweep always takes the plane; `a = 0` on air is the materials door, `tests/test_optics_ingress.py` |
| `test_glass_partially_attenuates_heat` | the march's exact `1 − 0.3` transmission | the sweep's transmission law is gate 0, bit for bit against `sweep_ref_q.py` |
| `test_wall_blocks_heat_past_it` | `heat_atten = 1` blocks downrange | subsumed by rewrite 1, which asserts the opaque end |
| `test_light_transparent_heat_opaque_tile` | the converse half of the independence | folded into rewrite 2 (both directions in one property) |
| `test_light_only_source_unaffected_by_heat_atten` | the 4-channel cull does not shorten a light ray | REMOVED: no rays, no per-channel cull. Light lands at P6 on this same sweep |
| `test_heat_only_survives_when_light_extinguished` | the same cull, other direction | REMOVED, as above |
| `test_heat_buffer_is_bit_identical_across_casts` | repeat-cast bit-identity | gate 0 + `tests/test_radiation_sweep_shadow_wiring.py::test_a_second_tick_overwrites_the_planes_and_does_not_accumulate` |
| `test_heat_atten_none_matches_zero_field` | the march's optional-plane API | REMOVED: the sweep's extinction planes are mandatory arguments — `tests/test_radiation_sweep_shadow_wiring.py::test_stale_caller_cannot_hand_step_tail_a_narrow_plane` |

Rewrite (2), exactly as §11.1 words them — *"the ordering (`air < glass < wall`)
and the heat-transparent/light-opaque independence become gate-3 scenes on the
sweep's extinction planes"*: `test_heat_ordering_air_glass_wall` and
`test_heat_transparent_light_opaque_tile`. See §2.3 and §2.4.

### 1.2 Group B — the `cool_shift` surface (L3 critique B3)

Not in design v3 §11.1; found by the L3 scope critique. These die with **R1**
(*"`cool_shift` is deleted"* — `docs/thermal_model_v1_design_2026-09-19.md` §2).

The channel that takes `cool_shift`'s job — and therefore the standing
replacement for every "a solid sheds heat to ambient" property below — is the
pair R1 names: **the sweep's `rad_amb` term** and **`k_leak` per tile**, both
live and gated today by
`tests/test_radiation_sweep_ambient_plane.py::test_item2_the_uniform_ambient_path_reproduces_the_scalar_era_integer_for_integer`,
`::test_item3_a_hull_wall_facing_vacuum_radiates_against_a_different_ambient`,
`::test_item4_conservation_is_exact_with_a_per_cell_ambient_and_the_leak_live`,
and `tests/test_radiation_sweep_gates.py::test_g1_three_term_identity_...`.

#### B1 · `tests/test_cool_shift_axis.py` — 25 functions / **47 collected** — **delete all**

Subject, from its own docstring: *"THE FIX (this patch). A per-material
`cool_shift` column, projected to the per-tile `GameMap.cool_shift` grid."* R1
deletes the column, the grid and the pass. (L3 counted 25 — the function count;
six are parametrized, so 47 collect.)

| group | fns | replacement |
|---|---|---|
| the column exists, is wired per row, defaults to the global, e-folds are the documented powers of two — `test_every_material_carries_the_column_wired_to_its_own_row`, `test_the_globals_are_kept_and_still_have_jobs`, `test_efold_seconds_are_the_documented_powers_of_two`, `test_column_is_optional_and_defaults_to_the_global` | 4 | REMOVED by R1. The per-tile loss dial that replaces it is `k_leak_q`, gated by `tests/test_radiation_sweep_ambient_plane.py` items 2–4 |
| loader validation of the column — `test_loader_accepts_the_legal_range`, `..._rejects_shift_zero_the_instant_total_wipe`, `..._rejects_below_the_floor`, `..._rejects_above_the_ceiling`, `..._rejects_non_integers`, `..._accepts_an_integer_valued_float` | 6 | REMOVED with the column. The ingress door for the plane that replaces it is `tests/test_optics_ingress.py` — `optics_fixed` RAISES outside [0, 1] (CLAUDE.md "Extinction planes") |
| the grid is the column projected on the `heat_inv_shift` seam, patched by `on_tile_changed`, in the resident set — `test_grid_is_the_table_column_projected`, `test_grid_is_built_in_the_same_function_as_heat_inv_shift`, `test_on_tile_changed_patches_the_grid_both_ways_and_is_O1`, `test_burning_out_a_crate_patches_the_grid`, `test_grid_joins_the_resident_mask_set` | 5 | REMOVED with the field. The seam itself — a plane is patched at `on_tile_changed` and joins residency — is held by `tests/test_gas_energy_tile_flip.py` (the membership seam, CLAUDE.md "Energy closure identity") and by `tests/test_fuel_fraction_axis.py`, the same projection pattern on a surviving plane |
| the decay law per tile: uniform equals scalar, per-tile honoured cell by cell, negatives round toward zero — `test_uniform_grid_at_the_seeded_value_equals_the_null_fallback`, `test_per_tile_shift_is_honoured_cell_by_cell`, `test_negative_temperatures_still_round_toward_zero_per_tile` | 3 | REMOVED by R1. Structurally the same three claims for the plane that replaces it are `tests/test_radiation_sweep_ambient_plane.py` item 2 (uniform == scalar era, integer for integer), item 3 (per-cell honoured — *breaks if the plane is read once and hoisted*) and item 4 (conservation) |
| the vacuum-offset rule and its floor — `test_vacuum_offset_reproduces_the_old_pair_at_the_seeded_value`, `test_vacuum_shift_is_the_base_minus_the_global_offset_floored`, `test_the_floor_binds_and_never_lets_the_exposed_shift_reach_zero`, `test_a_vacuum_exposed_tile_keeps_ONE_dial_per_material` | 4 | REMOVED by R1. "Space sheds faster" is now a property of the **ambient plane**, not of a shift: `::test_item3_the_cold_cell_is_the_cell_the_mask_names` and `::test_item3_derive_ambient_is_the_r3_derivation_and_keys_on_is_vacuum_alone` |
| two materials decay at different rates, in one grid, addressable per material — `test_two_materials_with_different_cool_shift_decay_at_different_rates`, `test_two_materials_in_ONE_grid_diverge_in_ONE_step`, `test_a_crate_grid_is_addressable_per_material` | 3 | REMOVED by R1. The surviving "per-tile, not hoisted" property is `::test_item3_a_hull_wall_facing_vacuum_radiates_against_a_different_ambient` |

#### B2 · `tests/test_temperature_cooling.py` — 9 collected — **delete all**

Subject: the Pass-3 ambient-cooling pass and the `COOL_SHIFT` /
`COOL_SHIFT_VACUUM` pair — the pass R1 deletes outright.

| test | what it pins | replacement |
|---|---|---|
| `test_shipped_cooling_dials` | the shipped dial pair | REMOVED with the dials |
| `test_hot_tile_relaxes_toward_zero_monotone` | Newtonian relaxation to ambient | REMOVED by R1 — and its *negation* is verification item 7 of `docs/thermal_model_v1_design_2026-09-19.md` §6 (*"Nothing relaxes to ambient any more"*), owned by the flip. The loss that replaces it is `rad_amb` + `k_leak`: G1 and ambient-plane item 4 |
| `test_vacuum_exposed_cools_about_4x_faster` | the vacuum shift | REMOVED; vacuum is now an ambient *level*, `::test_item3_*` |
| `test_low_atmosphere_neighbour_counts_as_exposed` | the atmosphere threshold flips the shift | REMOVED with the pass. The `is_vacuum` keying that replaces it is `::test_item3_derive_ambient_is_the_r3_derivation_and_keys_on_is_vacuum_alone` |
| `test_dead_band_settles_to_exact_rest` | the `T >> shift` dead band settles exactly | REMOVED with the shift. The sweep's exact-fixed-point equivalent is G2a: an ambient cell moves by exactly zero |
| `test_air_stays_bit_exactly_zero` | cooling skips non-solids | REMOVED with the pass. The sweep's thermal-solid mask is pinned by gate 0 and exercised by G2's mixed scenes |
| `test_negative_delta_relaxes_up_toward_zero` | symmetric relaxation from below ambient | REMOVED by R1. Sub-ambient cells stay in the gate vocabulary — `random_scene` in `tests/_radiation_sweep_harness.py` draws `T = −200` game, and G2a is asserted at ambient |
| `test_deterministic_bit_identical` | determinism of the pass | `tests/test_w6_armory.py:583` (GOLDEN_AGGREGATE) + gate 0 |
| `test_integration_inject_then_burn_out` | inject heat once; conduction + cooling bring it down | the conduction half survives untouched in `tests/test_temperature_conduction.py`; the cooling half is REMOVED by R1, and its post-flip successor is verification item 6 (*"A sealed room is no longer adiabatic"*), owned by the flip |

#### B3 · `tests/test_cuda_cool_shift.py` (1 collected, skipped here) + `tests/cuda_cool_shift_check.py` (check script, not collected) — **delete both**

The CUDA lockstep for the Pass-3 `temp_cool` kernel and its per-tile shift.
REMOVED with the kernel (R1). The pair goes together because
`test_cuda_cool_shift.py` exists only to run the check script through
`cuda_harness`. The CUDA temperature twin keeps its other tol-0 gates —
`tests/cuda_conduction_check.py` and `tests/cuda_thermal_mass_check.py` — and
the sweep's own GPU twin arrives with `cuda_radiation_sweep_check.py` at P4
(design v3 §11 P4, gate 7).

#### B4 · `tests/cool_shift_axis_gate_a_capture.py` — **NOT deleted. Finding, §5.1.**

### 1.3 Group C — D3, the R11 load-time sustain check

`src/simulation/materials.py::MaterialTable._check_ignition_seed` (method at
`:855`, call site at `:694`). Erik's R11: deleted — it asks an ill-posed
question and `cool_shift` guts it anyway. Its gain is literally
`bed_per_I · 2^(cool_shift − heat_inv_shift)` (`:952-958`).

**Nothing imports or asserts on it, and no test dies with it.** Verified by
`git grep` over all non-doc files: the only references outside `materials.py`
are two *comments* (`tools/fire_smother_curve_sweep.py:153`,
`tools/fire_tune_loop.py`'s derivation trail). It only ever
`print(..., file=sys.stderr)` — warning, never raising — so nothing can be gated
on it, and nothing is.

Dead with it and removed: `_COMB_DEFAULTS` (`:183`), `_SEED_CHECK_DT` (`:198`),
`_SEED_CHECK_CLAIM_FACES` (`:199`), `_X_AMBIENT` (`:204`), `_SEED_WARNED`
(`:209`). What is deliberately kept, and why: §5.3.

### 1.4 What the `cool_shift` grep turns up that is **not** mine

`git grep -il cool_shift -- ':!docs'` returns 43 files. The tests among them are
**partial hits** — they set the dial to disable cooling, or read `e_cool_sum` —
so they need an *edit* at the flip, not a deletion, and that edit is the flip
patch's: `tests/_phase3a_driver.py`, `cuda_conduction_check.py`,
`cuda_thermal_mass_check.py`, `fuel_fraction_axis_gate_a_capture.py`,
`o2_full_reference_gate_a_capture.py`, `test_eos_p2_sealed_room_energy.py`,
`test_eos_p4_combustion.py`, `test_fire_feedback.py`,
`test_fuel_fraction_axis.py`, `test_pr3_capacity_law.py`,
`test_radiation_sweep_gates.py`, `test_radiation_sweep_shadow_wiring.py`,
`test_temperature_conduction.py`, `test_temperature_convert.py`,
`test_thermal_mass_axis.py`, `test_thermostat_books.py`.

Out of scope by explicit instruction or another patch's ownership:

- `tests/test_thermal_mass_axis.py` — T3b already rewrote its snapshot into the
  R14 density property. Left alone.
- `tests/test_temperature_convert.py::test_air_tile_receives_radiation_deposit`
  — §11.1: **survives** (it exercises `heat`, not `rad_net`).
- `tests/test_cuda_pr1_fire_plane.py` + `cuda_pr1_fire_plane_check.py`,
  `tests/test_cuda_s2_raycaster.py` + `cuda_s2_check.py`,
  `cuda_s2b_raycaster_live_check.py`, `bench_s8c_fire_heat_check.py` +
  `test_s8c_fire_heat_bench.py` — disposed **at P4**, with
  `cuda_radiation_sweep_check.py` as the named replacement (design v3 §11 P4).
- all of §11.2, the render march (`test_heat_smoke_glow.py`,
  `test_multigas_colour.py`, `test_dyn_light_atten.py`,
  `test_rgb_light_atten.py`, `test_rgb_light_pack.py`, `test_fire_lights.py`) —
  disposed at **P6**.

### 1.5 The arithmetic

| group | file | collected | disposition |
|---|---|---|---|
| A1 | `test_pf1a_radiation_books.py` | 11 | delete file |
| A2 | `test_pr1_fire_plane_cast.py` | 11 | delete file |
| A3 | `test_fire_heat_source.py` | 14 | delete file (10 outright, 4 re-homed) |
| A4 | `test_heat_attenuation.py` | 10 | delete file (8 outright, 2 rewritten) |
| B1 | `test_cool_shift_axis.py` | 47 | delete file |
| B2 | `test_temperature_cooling.py` | 9 | delete file |
| B3 | `test_cuda_cool_shift.py` | 1 (skipped here — no CUDA build) | delete file + its check script |
| | **removed** | **103** | 102 currently *pass*, 1 *skips* |
| §2 | new pre-flip sweep tests | +4 | |

Predicted gate: **2631 − 102 + 4 = 2533 passed**, **29 − 1 = 28 skipped**,
4 xfailed, 0 failed.

---

## 2. D2 — the four new tests (filled in as they land)

## 3. D3 — the R11 sustain check (filled in when it lands)

## 4. The gate (filled in at the end)

## 5. Findings

### 5.1 `tests/cool_shift_axis_gate_a_capture.py` must NOT be deleted

L3's B3 table lists it as part of the `cool_shift` surface ("a committed gate-A
capture artifact for the axis"). It is not safe to delete, and the reason is the
exact silent-failure mode this patch exists to avoid:

- It is **not pytest-collected** (deliberately no `test_` prefix — its own
  docstring says so), so deleting it would leave the suite green.
- Two *later* arcs import its scenario library by name:
  `tests/fuel_fraction_axis_gate_a_capture.py:96` and
  `tests/o2_full_reference_gate_a_capture.py:134` both do
  `from cool_shift_axis_gate_a_capture import _SCENARIOS`, and
  `o2_full_reference_gate_a_capture.py:30` says so in its docstring
  (*"Scenarios are imported verbatim from cool_shift_axis_gate_a_capture"*).
  Neither importer is collected either, so both would break silently and stay
  broken.
- Its only `cool_shift` references are in the **docstring** (lines 4, 17, 22).
  The file is version-agnostic on purpose — *"it never touches gmap.cool_shift
  or any other new attribute"* — so R1 does not invalidate a line of its code.

Despite its name it is the shared before/after capture harness for three arcs.
Left alone. If the flip wants the name cleaned up, the move is a **rename** with
the two importers updated, never a delete.
