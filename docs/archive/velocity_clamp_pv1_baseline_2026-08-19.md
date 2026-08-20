# P-V1 step-0 baseline — pre-patch red-test list (2026-08-19)

Recorded before touching any code, on the untouched `velocity-clamp` tree
(HEAD `d9ae647`), per the P-V1 spec's gate 5 requirement. CUDA build was
rebuilt first (`cmd /c "cpp\build_cuda_lenovo.bat"` — "no work to do", already
current for this HEAD) so the CUDA gates exercise real code rather than
skipping.

Command: `conda run -n data python -m pytest tests -q` from repo root.

Result: **31 failed, 2239 passed, 5 skipped, 3 warnings** in 126.18s.

## Failing test name list (31)

```
tests/test_b1_signal_bus.py::test_dormancy_door_present_wire_free_digest_byte_identical
tests/test_b2_nodes.py::test_b1_dormancy_still_byte_identical
tests/test_b5_airlock.py::test_b1_dormancy_still_byte_identical
tests/test_b6_logic_golden.py::test_logic_loop_trajectory_digest_matches_committed_golden
tests/test_b6_logic_golden.py::test_logic_loop_digest_is_reproducible_and_2x_bit_identical
tests/test_b6_logic_golden.py::test_b1_dormancy_door_present_wire_free_still_byte_identical
tests/test_cool_shift_axis.py::test_every_material_carries_the_column_seeded_at_the_old_global
tests/test_cool_shift_axis.py::test_a_crate_grid_from_config_is_uniform_today_but_addressable
tests/test_cuda_p64_kick_compression.py::test_p64_kick_compression_bit_identity
tests/test_eos_p4_combustion.py::test_combustion_pass_conserves_o2_n2_soot_exactly
tests/test_eos_p4_combustion.py::test_e2e_1_sealed_room_fire_self_starves
tests/test_eos_p4_combustion.py::test_e2e_2_breach_vents_o2_and_kills_fire
tests/test_eos_p4_combustion.py::test_e2e_4_inert_flood_smothers_fire
tests/test_eos_p4_combustion.py::test_payoff_orderings_perturbation_robust
tests/test_eos_p5_1_stoich.py::test_fuel_decrement_exact_and_deterministic
tests/test_eos_p5_1_stoich.py::test_one_lsb_floor_never_crossed
tests/test_eos_p5_1_stoich.py::test_no_destruction_originates_from_combustion
tests/test_eos_p5_1_stoich.py::test_lifecycle_ember_reignite_charout
tests/test_eos_p6_9_isotropy.py::test_isotropy_bit_exact_zero_remainder
tests/test_eos_p6_9_isotropy.py::test_isotropy_bounded_bias_nonzero_remainder
tests/test_fire_feedback.py::test_cold_fire_decays_to_zero
tests/test_fire_feedback.py::test_low_o2_fire_decays_to_zero
tests/test_fire_feedback.py::test_vented_room_extinguishes
tests/test_fire_feedback.py::test_burnout_when_wall_hp_runs_out
tests/test_fire_feedback.py::test_wind_blows_out_a_small_fire
tests/test_fire_feedback.py::test_plume_raises_own_atmosphere_wind_points_outward
tests/test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood
tests/test_fire_o2_invariant.py::test_production_ignition_matches_cpp_gate_off_tie
tests/test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing
tests/test_pr3_capacity_law.py::test_fire_T_ext_is_derived_from_ignition_temp
tests/test_s3b_fire_determinism.py::test_fire_field_and_burnthrough_list_bit_identical_run_twice
```

Note: this is fire/combustion/logic-golden/digest debt, not velocity-clamp
related — consistent with the spec's "expect ~dozens of standing reds
including six digest tests" framing (six digest-named tests are present in
this list: the two `test_b1_dormancy_still_byte_identical` copies, the three
`test_b6_logic_golden` tests, and `test_cuda_p64_kick_compression::test_p64_kick_compression_bit_identity`).

Gate 5 (post-patch) measures the full suite against this exact 31-name list:
zero NEW red beyond it, except the pre-declared GOLDEN_AGGREGATE flip set
(measured separately once the clamp is wired in).
