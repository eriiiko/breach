"""
Breach physics engine -- C++ accelerated simulation
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['ATMOSPHERE_FIXEDPOINT', 'ATMOSPHERE_FP_ONE', 'ATMOSPHERE_FP_SHIFT', 'AtmosphereSolver', 'CAP_SHIFT_MAX', 'CAP_SHIFT_MIN', 'CombustionSolver', 'EDIT_FIELD_ORDINALS', 'EDIT_QUEUE_LAYOUT', 'EOSSolver', 'E_CEILING_TOP_Q', 'E_FINE_BITS', 'E_INV_TOP_GAME', 'E_TABLE_SIZE', 'E_TABLE_TOP_MAX', 'EditDoorBuffer', 'EditFlushBuffer', 'EmissiveTable', 'FieldApplierBuffer', 'FireParams', 'FireSimulation', 'HAS_CUDA', 'L_CHANNELS', 'L_FINE_BITS', 'L_TABLE_TOP_MAX', 'LightEmissionTable', 'ObjectTableBuffer', 'PHILOX32_TWO_PI_Q16', 'PhysicsEngine', 'RADIATION_SWEEP_CNT_SLOTS', 'RS_BAD_HEAT_ABSORB', 'RS_SLOT_BAD_AMBIENT', 'RS_SLOT_BAD_EXTINCTION', 'RS_SLOT_BAD_LIGHT_EXTINCTION', 'RS_SLOT_BAD_SCALARS', 'RS_SLOT_LIGHT_ABSORB', 'RS_SLOT_LIGHT_EMIT', 'RS_SLOT_LIGHT_MAX_STREAM', 'RS_SLOT_LIGHT_MIN_STREAM', 'RS_SLOT_LIGHT_RING_IN', 'RS_SLOT_LIGHT_RING_OUT', 'RS_SLOT_MAX_STREAM', 'RS_SLOT_MIN_STREAM', 'RadiationSweep', 'SWARM_CONSTANTS', 'SWARM_LARVA_BRANCHES', 'SWARM_SPECIES_LAWS', 'SWARM_TABLES_FINGERPRINT', 'TRACE_CELL_MAX', 'TRACE_FINE_BITS', 'TRACE_SHIFT', 'TemperatureSolver', 'UnitEventRingBuffer', 'UnitRecordListBuffer', 'UnitStoreBuffer', 'UnitTablesBuffer', 'WATER_FIXEDPOINT', 'WATER_FP_ONE', 'WATER_FP_SHIFT', 'WAVE_FIXEDPOINT', 'WAVE_FP_ONE', 'WAVE_FP_SHIFT', 'WIND_FIXEDPOINT', 'WIND_FP_ONE', 'WIND_FP_SHIFT', 'WaterSolver', 'applier_blast_probe', 'applier_layout', 'applier_run', 'atan2_q16', 'build_id', 'bulk_flux_transport', 'conduction_cell_capacity_q', 'conductor_books_begin', 'conductor_books_deferring', 'conductor_books_end', 'conductor_label', 'cos_q16', 'cuda_available', 'cuda_bulk_flux_transport', 'cuda_combustion_step', 'cuda_device_info', 'cuda_eos_energy_flux', 'cuda_eos_kick_compression', 'cuda_eos_mg_solve', 'cuda_eos_sl_advect', 'cuda_fire_step', 'cuda_map_mul_q16', 'cuda_radiation_sweep_launch_count', 'cuda_radiation_sweep_resident', 'cuda_radiation_sweep_run', 'cuda_spike_add1', 'cuda_swarm_resident_probe', 'cuda_swarm_species_step', 'cuda_temperature_step', 'cuda_water_step', 'edit_door_append_probe', 'edit_door_layout', 'edit_flush_layout', 'edit_queue_contrib_probe', 'eos_energy_books_sum', 'eos_kick_compression_ref', 'eos_mg_build_parity', 'eos_mg_solve_ref', 'eos_nsub_reads', 'eos_prestage_parity', 'eos_resident_calls', 'eos_sl_advect_ref', 'eos_step_cuda_calls', 'fp_deposit_dT_wide_i64', 'fp_deposit_dT_wide_q16', 'fp_div_round', 'fp_div_trunc', 'fp_effort_step', 'fp_falloff_floor', 'fp_falloff_round', 'fp_fine_heat_shr', 'fp_floordiv', 'fp_half_up', 'fp_isqrt', 'fp_isqrt_ceil', 'fp_isqrt_u128', 'fp_isqrt_u64', 'fp_make_recip', 'fp_mul_q16_round', 'fp_muldiv_floor', 'fp_quantize', 'fp_quantize_trace', 'fp_recip_mul', 'fp_reciprocal_q16', 'fp_shr_round0', 'fp_shr_round0_i64', 'fp_shr_round0_signed_i64', 'fp_shr_trunc', 'fp_sqrt_q16', 'fp_trace_add_sat', 'fp_trace_density_shr', 'fp_trace_from_bulk', 'get_bulk_flux_backend', 'get_combustion_backend', 'get_eos_step_backend', 'get_fire_backend', 'get_kick_compression_backend', 'get_mg_solve_backend', 'get_radiation_backend', 'get_sl_advection_backend', 'get_swarm_backend', 'get_temperature_backend', 'get_water_backend', 'horde_flood_probe', 'host_sync_log_begin', 'host_sync_log_end', 'object_tables_layout', 'philox32_4x32_10', 'philox32_4x32_10_np', 'philox32_angle_q16', 'philox32_below', 'philox32_uniform_q16', 'radiation_sweep_cuda_calls', 'resident_burn_list', 'resident_capture_abort', 'resident_capture_begin', 'resident_capture_end', 'resident_ignition_probe', 'resident_stage_counters', 'set_bulk_flux_backend', 'set_combustion_backend', 'set_fire_backend', 'set_kick_compression_backend', 'set_mg_solve_backend', 'set_radiation_backend', 'set_sl_advection_backend', 'set_swarm_backend', 'set_temperature_backend', 'set_water_backend', 'sin_q16', 'sky_exchange_step', 'swarm_branch_tally', 'swarm_cpu_calls', 'swarm_cuda_calls', 'swarm_larva_felt_T', 'swarm_species_step', 'swarm_wrap_heading_q16', 'tick_end_clears', 'tick_stream_ptr', 'topology_probe_seam', 'topology_probe_tile_changed', 'unit_apply_orders', 'unit_burn_sort', 'unit_burst_walls', 'unit_charges_tick', 'unit_convert_zombies', 'unit_damage_probe', 'unit_door_uses', 'unit_draws_layout', 'unit_events_layout', 'unit_expire_orders', 'unit_exposure_decay', 'unit_flight_plane', 'unit_gas_rows', 'unit_grenades', 'unit_heat_damage', 'unit_ignition', 'unit_integrate_turn', 'unit_kit_draws_probe', 'unit_kit_falloff_probe', 'unit_kit_sight_probe', 'unit_lib_probe', 'unit_match_done', 'unit_melee', 'unit_movement', 'unit_movement_probe', 'unit_requests_layout', 'unit_rounds', 'unit_shooting', 'unit_spray_cone_probe', 'unit_sprays', 'unit_stamp', 'unit_status_tick', 'unit_store_layout', 'unit_table_appends', 'unit_verbs', 'unit_verbs_probe', 'unit_verbs_tables', 'unit_wind_impulse', 'water_substeps_resident']
class AtmosphereSolver:
    def __init__(self) -> None:
        ...
    def diffuse_solve(self, atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_p: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_v: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_source: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def gs_residual(self) -> float:
        ...
    def max_dt(self) -> float:
        ...
    def max_dt_q(self) -> int:
        ...
    def step(self, wave_p: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_v: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_source: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def wave_substep(self, wave_p: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_v: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_source: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def absorb_strength(self) -> float:
        ...
    @absorb_strength.setter
    def absorb_strength(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def breach_rate(self) -> float:
        ...
    @breach_rate.setter
    def breach_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c(self) -> float:
        ...
    @c.setter
    def c(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d_atm(self) -> float:
        ...
    @d_atm.setter
    def d_atm(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def damping(self) -> float:
        ...
    @damping.setter
    def damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def feed_rate(self) -> float:
        ...
    @feed_rate.setter
    def feed_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gs_iters(self) -> int:
        ...
    @gs_iters.setter
    def gs_iters(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def last_gs_residual(self) -> float:
        ...
    @property
    def max_source_per_step(self) -> float:
        ...
    @max_source_per_step.setter
    def max_source_per_step(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def transfer(self) -> float:
        ...
    @transfer.setter
    def transfer(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CombustionSolver:
    def __init__(self) -> None:
        ...
    def step(self, gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], o2_idx: typing.SupportsInt | typing.SupportsIndex, inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, black_smoke_idx: typing.SupportsInt | typing.SupportsIndex, temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], ignition_temp_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dt: typing.SupportsFloat | typing.SupportsIndex, c_v: typing.SupportsFloat | typing.SupportsIndex, n_floor_heat: typing.SupportsFloat | typing.SupportsIndex, thermal_solid: typing.Any = None, heat_inv_shift: typing.Any = None, heat: typing.Any = None, dem_acc: typing.Any = None, draw_r: typing.SupportsInt | typing.SupportsIndex = 1, dyn_permeability: typing.Any = None, max_claimants: typing.SupportsInt | typing.SupportsIndex = 4, gas_energy: typing.Any = None, is_ambient: typing.Any = None, t_amb_q: typing.SupportsInt | typing.SupportsIndex = 0, fire_T_ext_plane: typing.Any = None, fuel_per_o2_plane: typing.Any = None, atmosphere: typing.Any = None, eos: typing.Any = None) -> None:
        ...
    @property
    def H_BED_M(self) -> float:
        ...
    @H_BED_M.setter
    def H_BED_M(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def H_BED_SHIFT(self) -> int:
        ...
    @H_BED_SHIFT.setter
    def H_BED_SHIFT(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def H_FUEL_M(self) -> float:
        ...
    @H_FUEL_M.setter
    def H_FUEL_M(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def H_FUEL_SHIFT(self) -> int:
        ...
    @H_FUEL_SHIFT.setter
    def H_FUEL_SHIFT(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def T_MAX_PHYS(self) -> float:
        ...
    @T_MAX_PHYS.setter
    def T_MAX_PHYS(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def burn_rate(self) -> float:
        ...
    @burn_rate.setter
    def burn_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def e_comb_deliver_sum(self) -> int:
        ...
    @property
    def e_comb_draw_sum(self) -> int:
        ...
    @property
    def e_comb_export_sum(self) -> int:
        ...
    @property
    def e_comb_heat_sum(self) -> int:
        ...
    @property
    def e_comb_mint_sum(self) -> int:
        ...
    @property
    def e_comb_rail_sum(self) -> int:
        ...
    @property
    def e_comb_solid_heat_sum(self) -> int:
        ...
    @property
    def e_deposit_drop_sum(self) -> int:
        ...
    @property
    def e_soot_shed_sum(self) -> int:
        ...
    @property
    def e_ts_products_sum(self) -> int:
        ...
    @property
    def fire_T_ext(self) -> float:
        ...
    @fire_T_ext.setter
    def fire_T_ext(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fire_T_span(self) -> float:
        ...
    @fire_T_span.setter
    def fire_T_span(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fuel_per_o2(self) -> float:
        ...
    @fuel_per_o2.setter
    def fuel_per_o2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def heat_floor_hits(self) -> int:
        ...
    @property
    def hotf_cap(self) -> float:
        ...
    @hotf_cap.setter
    def hotf_cap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_frac_amb(self) -> float:
        ...
    @o2_frac_amb.setter
    def o2_frac_amb(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_frac_ext(self) -> float:
        ...
    @o2_frac_ext.setter
    def o2_frac_ext(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_frac_full(self) -> float:
        ...
    @o2_frac_full.setter
    def o2_frac_full(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_thresh_breathe(self) -> float:
        ...
    @o2_thresh_breathe.setter
    def o2_thresh_breathe(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_thresh_burn(self) -> float:
        ...
    @o2_thresh_burn.setter
    def o2_thresh_burn(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def p_ext_q(self) -> int:
        ...
    @p_ext_q.setter
    def p_ext_q(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def p_full_q(self) -> int:
        ...
    @p_full_q.setter
    def p_full_q(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def soot_yield(self) -> float:
        ...
    @soot_yield.setter
    def soot_yield(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def t_max_phys_hits(self) -> int:
        ...
class EOSSolver:
    debug_pstar_from_prev: bool
    use_multigrid: bool
    def __init__(self) -> None:
        ...
    def boundary_flux(self) -> list:
        ...
    def dbg_mg_inputs(self) -> tuple[numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.int32], numpy.typing.NDArray[numpy.int32]]:
        """
        P6.3: (pstar, div_u, n_total) flat int32 copies as consumed by the last step()'s pressure solve.
        """
    def reset_trace_sat_books(self, n_gases: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        #88 P2: zero trace_clip_sum / trace_mint_sum / trace_neg_hits (sized n_gases) -- PhysicsRunner's per-tick reset.
        """
    def set_div_source(self, arg0: typing.Any) -> None:
        """
        #31: this tick's (h,w) int32 Q16 1/s divergence source, or None.
        """
    def trace_clip_sum(self) -> list:
        ...
    def trace_decay_sum(self) -> list:
        ...
    def trace_mint_sum(self) -> list:
        ...
    def trace_neg_hits(self) -> list:
        ...
    def trace_sink_sum(self) -> list:
        ...
    def trace_wipe_sum(self) -> list:
        ...
    @property
    def C(self) -> float:
        ...
    @C.setter
    def C(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def CFL_ADV(self) -> float:
        ...
    @CFL_ADV.setter
    def CFL_ADV(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def N_FLOOR_SOLVER(self) -> float:
        ...
    @N_FLOOR_SOLVER.setter
    def N_FLOOR_SOLVER(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def N_SUB_MAX(self) -> int:
        ...
    @N_SUB_MAX.setter
    def N_SUB_MAX(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def S(self) -> int:
        ...
    @S.setter
    def S(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def S_EOS(self) -> float:
        ...
    @S_EOS.setter
    def S_EOS(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def T_AMB_K(self) -> float:
        ...
    @T_AMB_K.setter
    def T_AMB_K(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def T_MAX_PHYS(self) -> float:
        ...
    @T_MAX_PHYS.setter
    def T_MAX_PHYS(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def T_MIN(self) -> float:
        ...
    @T_MIN.setter
    def T_MIN(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def U_MAX(self) -> float:
        ...
    @U_MAX.setter
    def U_MAX(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def absorb_strength(self) -> float:
        ...
    @absorb_strength.setter
    def absorb_strength(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def adiabatic_index(self) -> float:
        ...
    @adiabatic_index.setter
    def adiabatic_index(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_max(self) -> float:
        ...
    @c_max.setter
    def c_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_v(self) -> float:
        ...
    @c_v.setter
    def c_v(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dbg_T_post_advect(self) -> int:
        ...
    @property
    def dbg_T_post_compression(self) -> int:
        ...
    @property
    def dbg_T_pre_advect(self) -> int:
        ...
    @property
    def dbg_last_c_local_q(self) -> int:
        ...
    @property
    def dbg_last_n_sub(self) -> int:
        ...
    @property
    def dbg_probe_idx(self) -> int:
        ...
    @dbg_probe_idx.setter
    def dbg_probe_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def digest_advect(self) -> int:
        ...
    @property
    def digest_bulk_flux(self) -> int:
        ...
    @property
    def digest_compression(self) -> int:
        ...
    @property
    def digest_helmholtz(self) -> int:
        ...
    @property
    def digest_pstar(self) -> int:
        ...
    @property
    def digest_velocity(self) -> int:
        ...
    @property
    def dx(self) -> float:
        ...
    @dx.setter
    def dx(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def e_absorb_export_sum(self) -> int:
        ...
    @property
    def e_clamp_destroyed_sum(self) -> int:
        ...
    @property
    def e_drag_heat_sum(self) -> int:
        ...
    @property
    def e_energy_floor_sum(self) -> int:
        ...
    @property
    def e_entry_resync_sum(self) -> int:
        ...
    @property
    def e_floor_sum(self) -> int:
        ...
    @property
    def e_kick_ke_sum(self) -> int:
        ...
    @property
    def e_rail_sum(self) -> int:
        ...
    @property
    def e_retire_sum(self) -> int:
        ...
    @property
    def e_sponge_export_sum(self) -> int:
        ...
    @property
    def e_transport_net_sum(self) -> int:
        ...
    @property
    def e_ts_ke_sum(self) -> int:
        ...
    @property
    def e_ts_residual(self) -> int:
        ...
    @property
    def e_ts_work_sum(self) -> int:
        ...
    @property
    def e_wall_work_probe_sum(self) -> int:
        ...
    @property
    def e_wipe_sum(self) -> int:
        ...
    @property
    def e_work_export_sum(self) -> int:
        ...
    @property
    def energy_floor_hits(self) -> int:
        ...
    @property
    def eth_compression_delta(self) -> int:
        ...
    @property
    def eth_transport_delta(self) -> int:
        ...
    @property
    def flux_sat_hits(self) -> int:
        ...
    @property
    def k_drag(self) -> float:
        ...
    @k_drag.setter
    def k_drag(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_drag2(self) -> float:
        ...
    @k_drag2.setter
    def k_drag2(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ke_drag_removed(self) -> int:
        ...
    @property
    def mg_coarsest_sweeps(self) -> int:
        ...
    @mg_coarsest_sweeps.setter
    def mg_coarsest_sweeps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def mg_cycles(self) -> int:
        ...
    @mg_cycles.setter
    def mg_cycles(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def mg_min_dim(self) -> int:
        ...
    @mg_min_dim.setter
    def mg_min_dim(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def mg_nu1(self) -> int:
        ...
    @mg_nu1.setter
    def mg_nu1(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def mg_nu2(self) -> int:
        ...
    @mg_nu2.setter
    def mg_nu2(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def n_active_flux(self) -> int:
        ...
    @property
    def n_bulk_active_sum(self) -> int:
        ...
    @property
    def p_face_ceil_hits(self) -> int:
        ...
    @property
    def p_face_floor_hits(self) -> int:
        ...
    @property
    def rad_clip_hits(self) -> int:
        ...
    @property
    def t_max_phys_hits(self) -> int:
        ...
    @property
    def u_clamp_hits(self) -> int:
        ...
    @property
    def u_max_hits(self) -> int:
        ...
    @property
    def work_clamp_hits(self) -> int:
        ...
class EditDoorBuffer:
    """
    #29 P7a.3: the device edit door's per-env EDIT and TILE rows and seq counters (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, edit_cap: typing.SupportsInt | typing.SupportsIndex, tile_cap: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def begin(self) -> None:
        """
        The 6b flush's reset: every env's rows empty, its seq 0.
        """
    def reset_env(self, env: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        #29 P7e.3: env e's reset alone: its rows empty, its seq 0.
        """
    def seq(self) -> numpy.typing.NDArray[numpy.int64]:
        ...
    def set_seq(self, seq: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        ...
    @property
    def edits(self) -> UnitRecordListBuffer:
        ...
    @property
    def tiles(self) -> UnitRecordListBuffer:
        ...
class EditFlushBuffer:
    """
    #29 P7e.3: slot 6b's flush buffers built on the device from the edit door (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, edit_cap: typing.SupportsInt | typing.SupportsIndex, tile_cap: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def books(self) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.uint64]]:
        """
        THE ONE READ: (books (N, EDQ_SLOTS), fault words (N,), ~0 none).
        """
    def rows(self) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
        """
        The gate's download: (edits, tiles, env offsets, env scalars).
        """
    def run(self, door: EditDoorBuffer, tables: UnitTablesBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex, planes: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, o2_idx: typing.SupportsInt | typing.SupportsIndex, n2_idx: typing.SupportsInt | typing.SupportsIndex, salt: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Build from the door, K1 / K2, the door's reset. Downloads nothing.
        """
    def run_host_pack(self, buf: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], n_edits: typing.SupportsInt | typing.SupportsIndex, n_tiles: typing.SupportsInt | typing.SupportsIndex, planes: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, o2_idx: typing.SupportsInt | typing.SupportsIndex, n2_idx: typing.SupportsInt | typing.SupportsIndex, salt: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        THE GATE'S REFERENCE: K1 / K2 fed from a host pack (_edit_buffer).
        """
    @property
    def edit_cap(self) -> int:
        ...
    @property
    def n_env(self) -> int:
        ...
    @property
    def tile_cap(self) -> int:
        ...
class EmissiveTable:
    def __init__(self) -> None:
        ...
    def bake(self) -> None:
        """
        Bake (or re-bake) the E° table from the current dials -- in the currency `fine_bits` (rad_scale * 2^fine_bits; [0, E_FINE_BITS], ValueError outside it).
        """
    def e_bucket_of(self, T_q: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        emissive_table.h e_bucket_of: Q16.16 temperature -> bucket index.
        """
    def e_ceiling_q(self, phi: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        emissive_table.h e_ceiling_q: the maximum-principle clamp's ceiling as a Q16.16 game temperature -- the TOP (last Q16 value) of the first bucket whose E° exceeds Φ; 0 below E°[0]; saturates one LSB below 16000 game (E_CEILING_TOP_Q), below T_MAX_PHYS.
        """
    def e_inv_q(self, phi: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        emissive_table.h e_inv_q: E°⁻¹(Φ) as a Q16.16 game temperature (the bucket's LOW edge; 0 below E°[0]; saturates at 15996 game). The radiation temperature -- not the clamp's ceiling (e_ceiling_q).
        """
    def table(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        A COPY of the baked E° table (E_TABLE_SIZE int64 entries); bakes on first use and whenever a dial has moved.
        """
    @property
    def fine_bits(self) -> int:
        ...
    @fine_bits.setter
    def fine_bits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def k_temp_to_kelvin(self) -> float:
        ...
    @k_temp_to_kelvin.setter
    def k_temp_to_kelvin(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def kelvin_ambient(self) -> float:
        ...
    @kelvin_ambient.setter
    def kelvin_ambient(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rad_scale(self) -> float:
        ...
    @rad_scale.setter
    def rad_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FieldApplierBuffer:
    """
    #29 P6.4b / P7a.1: the per-env EFFECT LIST (N, cap, F) -- a record list with per-row staging for an n_units-row store -- and book channels (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, cap: typing.SupportsInt | typing.SupportsIndex, n_units: typing.SupportsInt | typing.SupportsIndex = 1, row_cap: typing.SupportsInt | typing.SupportsIndex = 32) -> None:
        ...
    def begin(self) -> None:
        """
        Empty every env's list (tick stream).
        """
    def download(self) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
        """
        (records (N, cap, F), counts (N,), dropped (N,)) int64.
        """
    def stage(self, recs: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], counts: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        ...
    @property
    def shape(self) -> tuple[int, int, int]:
        ...
class FireParams:
    def __init__(self) -> None:
        ...
    @property
    def I_cap_per_avail(self) -> float:
        ...
    @I_cap_per_avail.setter
    def I_cap_per_avail(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def I_min(self) -> float:
        ...
    @I_min.setter
    def I_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def P_full(self) -> float:
        ...
    @P_full.setter
    def P_full(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def P_min(self) -> float:
        ...
    @P_min.setter
    def P_min(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fire_T_ext(self) -> float:
        ...
    @fire_T_ext.setter
    def fire_T_ext(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fire_T_span(self) -> float:
        ...
    @fire_T_span.setter
    def fire_T_span(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def fuel_ref(self) -> float:
        ...
    @fuel_ref.setter
    def fuel_ref(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def hotf_cap(self) -> float:
        ...
    @hotf_cap.setter
    def hotf_cap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_die(self) -> float:
        ...
    @k_die.setter
    def k_die(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_grow(self) -> float:
        ...
    @k_grow.setter
    def k_grow(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_wind_fan(self) -> float:
        ...
    @k_wind_fan.setter
    def k_wind_fan(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_wind_strip(self) -> float:
        ...
    @k_wind_strip.setter
    def k_wind_strip(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_frac_amb(self) -> float:
        ...
    @o2_frac_amb.setter
    def o2_frac_amb(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_frac_ext(self) -> float:
        ...
    @o2_frac_ext.setter
    def o2_frac_ext(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_frac_full(self) -> float:
        ...
    @o2_frac_full.setter
    def o2_frac_full(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def o2f_cap(self) -> float:
        ...
    @o2f_cap.setter
    def o2f_cap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def p_expand_ref(self) -> float:
        ...
    @p_expand_ref.setter
    def p_expand_ref(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def p_ext_q(self) -> int:
        ...
    @p_ext_q.setter
    def p_ext_q(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def p_full_q(self) -> int:
        ...
    @p_full_q.setter
    def p_full_q(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def temp_scale(self) -> float:
        ...
    @temp_scale.setter
    def temp_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wall_damage(self) -> float:
        ...
    @wall_damage.setter
    def wall_damage(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class FireSimulation:
    params: FireParams
    def __init__(self) -> None:
        ...
    def step(self, fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_o2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, fuel_recip: typing.Any = None, fire_T_ext_plane: typing.Any = None) -> list:
        ...
    @property
    def dbg_probe_idx(self) -> int:
        ...
    @dbg_probe_idx.setter
    def dbg_probe_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class LightEmissionTable:
    fine_bits: typing.ClassVar[int] = 37
    def __init__(self) -> None:
        ...
    def table(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        A COPY of the checked-in L° as int64 (3, E_TABLE_SIZE), channel-major (R, G, B) over E°'s buckets.
        """
class ObjectTableBuffer(UnitRecordListBuffer):
    """
    #29 P7.0b: a per-env device object table (N, cap, width) int64 with counts and overflow counts (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, cap: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, live: typing.SupportsInt | typing.SupportsIndex, n_units: typing.SupportsInt | typing.SupportsIndex = 1, row_cap: typing.SupportsInt | typing.SupportsIndex = 1) -> None:
        ...
    def append(self, recs: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], n: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        """
        Append each env's first n[e] records in order (past cap: counted).
        """
    def compact(self) -> None:
        """
        The stable compaction of every env (rows whose live column is 0 dropped).
        """
    def upload(self, rows: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], counts: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        """
        Set every env's rows and count; the overflow counts back to 0.
        """
    @property
    def live(self) -> int:
        ...
class PhysicsEngine:
    def __init__(self) -> None:
        ...
    def apply_edits_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, buf: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], n_edits: typing.SupportsInt | typing.SupportsIndex, n_tiles: typing.SupportsInt | typing.SupportsIndex, salt: typing.SupportsInt | typing.SupportsIndex, o2_idx: typing.SupportsInt | typing.SupportsIndex, n2_idx: typing.SupportsInt | typing.SupportsIndex, n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None) -> None:
        """
        #29 P4c: the device edit queue -- every env's packed FieldEdit records (field_edit.pack_for_device) applied on the resident planes (cuda_edit_queue.cu); the books fold into edit_queue_books at step_tail_resident.
        """
    def gas_capacity_q(self) -> tuple[int, int, int]:
        """
        (n_floor_q, c_v_q, recip_cv): THE GAS CURRENCY — quantize(temperature.n_floor_heat), quantize(temperature.c_v) and make_recip(c_v_q / 65536), the integers the temperature fold's gas deposit divides by and the sweep's gas Fleck arm prices a gas cell in (P5b).
        """
    def horde_field(self, tile_ok: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], fp: typing.SupportsInt | typing.SupportsIndex, seeds: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], out: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> int:
        ...
    def relight(self, temperature: numpy.typing.NDArray[numpy.int32], heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], heat_inv_shift: numpy.typing.NDArray[numpy.int32], thermal_solid: numpy.typing.NDArray[numpy.bool], is_vacuum: numpy.typing.NDArray[numpy.bool], gas: numpy.typing.NDArray[numpy.int32], gas_conservative: numpy.typing.NDArray[numpy.bool], gas_heat_absorb_q16: numpy.typing.NDArray[numpy.int32], k_leak_q: typing.SupportsInt | typing.SupportsIndex, rad_amb_vacuum_q: typing.SupportsInt | typing.SupportsIndex, t_amb_q: typing.SupportsInt | typing.SupportsIndex, light_atten_q: typing.Any, dyn_light_atten_q: typing.Any, light_q: typing.Any, light_flux_q: typing.Any, light_glow: typing.Any, gas_light_absorb_q16: typing.Any = None, gas_light_glow_q16: typing.Any = None, light_cones: typing.Any = None, light_sky: typing.Any = None, light_emit_mult: typing.Any = None) -> None:
        """
        P6b: recompute the light field on the CURRENT state without a tick (the paused renderer's path): the one sweep invocation step_tail uses, light always on, its heat outputs into the engine's own scratch -- no plane a tick reads is written. Writes light_q / light_flux_q / light_glow in place.
        """
    def run_combustion_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, acc_depth: typing.SupportsInt | typing.SupportsIndex, o2_idx: typing.SupportsInt | typing.SupportsIndex, inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, black_smoke_idx: typing.SupportsInt | typing.SupportsIndex, draw_r: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ignition_table: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], ambient: bool, t_amb_q: typing.SupportsInt | typing.SupportsIndex, n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None) -> None:
        ...
    def run_substeps(self, p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_energy: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_diffusion: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas_decay: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, is_ambient: typing.Any = None, n_amb: typing.Any = None, p_amb: typing.SupportsInt | typing.SupportsIndex = 0, sponge_sigma: typing.Any = None, sponge_udamp: typing.Any = None, do_traces: bool = True, thermal_solid: typing.Any = None) -> None:
        ...
    def run_substeps_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], sim_time: typing.SupportsFloat | typing.SupportsIndex, ambient: bool = False, n_amb: typing.Any = None, p_amb: typing.SupportsInt | typing.SupportsIndex = 0, n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None) -> int:
        """
        #29 P5: the EOS stage over the batch; returns max(n_sub).
        """
    def run_trace_tail_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas_diffusion: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_decay: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], sim_time: typing.SupportsFloat | typing.SupportsIndex, ambient: bool = False, n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None) -> None:
        ...
    def sight_rays(self, solid: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], screen: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], physical: bool, pairs: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], out: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> None:
        ...
    def sight_window(self, solid: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], screen: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], physical: bool, y: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsInt | typing.SupportsIndex, wy: typing.SupportsInt | typing.SupportsIndex, wx: typing.SupportsInt | typing.SupportsIndex, to_window: bool, out: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8]) -> None:
        ...
    def stamp_units(self, permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], ys: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], xs: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], perm: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], wabsorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], heat_q: numpy.typing.NDArray[numpy.int32], light_atten_q: numpy.typing.NDArray[numpy.int32], dyn_light_atten_q: numpy.typing.NDArray[numpy.int32], light_q_rows: numpy.typing.NDArray[numpy.int32]) -> None:
        ...
    def stamp_units_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, rows: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None) -> None:
        """
        #29 P4b: the five stamp outputs on the resident planes from the host stamps' rows (cuda_stamp.cu); #29 P5: every env's rows, the env column the batch index.
        """
    def step_ripple(self, ripple: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], ripple_v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], sim_time: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        #29 P4a: the W6a ripple (the guard + WaterSolver.step_ripple), step_tail's step 1 as its own entry.
        """
    def step_swarm(self, packs: collections.abc.Sequence, temperature: typing.Any, solid: typing.Any, env: typing.Any, host_dirty: dict, bite: typing.Any, count: typing.Any) -> numpy.typing.NDArray[numpy.int32]:
        """
        arc #63 P3: the host-path swarm step (P5a / P5b: rebuilds the bite and count planes); returns the death rows (n, ROW_WORDS) int32 in (species, env, slot) order ((0, ROW_WORDS) when dormant), and sets host_dirty[sp] after a launch.
        """
    def step_tail(self, ripple: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], ripple_v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_p: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], heat: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat_inv_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], face_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], thermal_solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], fuel_recip: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], fire_T_ext_plane: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], o2_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], rad_net_sweep: numpy.typing.NDArray[numpy.int64], rad_flux_sweep: numpy.typing.NDArray[numpy.int64], rad_amb_sweep: numpy.typing.NDArray[numpy.int64], rad_fluence: numpy.typing.NDArray[numpy.int64], gas_heat_absorb_q16: numpy.typing.NDArray[numpy.int32], k_leak_q: typing.SupportsInt | typing.SupportsIndex, rad_amb_vacuum_q: typing.SupportsInt | typing.SupportsIndex = -1, is_ambient: typing.Any = None, rad_net: typing.Any = None, gas_energy: typing.Any = None, t_amb_q: typing.SupportsInt | typing.SupportsIndex = 0, light_atten_q: typing.Any = None, dyn_light_atten_q: typing.Any = None, light_q: typing.Any = None, light_flux_q: typing.Any = None, light_glow: typing.Any = None, gas_light_absorb_q16: typing.Any = None, gas_light_glow_q16: typing.Any = None, light_cones: typing.Any = None, light_sky: typing.Any = None, light_emit_mult: typing.Any = None) -> list:
        ...
    def step_tail_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], o2_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ambient: bool, t_amb_q: typing.SupportsInt | typing.SupportsIndex, k_leak_q: typing.SupportsInt | typing.SupportsIndex, rad_amb_vacuum_q: typing.SupportsInt | typing.SupportsIndex, gas_heat_absorb_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_light_absorb_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_light_glow_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None, swarm: typing.Any = None) -> typing.Any:
        """
        #29 P5: the tail + the ONE books D2H; returns each env's burn-through list (a list of N lists, row-major). arc #63 P3: with swarm= (swarm.engine_args(..., resident=True)) the swarm stage runs last and the return is (burn-through lists, per-env death rows).
        """
    def step_water(self, water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vx: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vy: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], floor_height: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], before: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], steam_idx: typing.SupportsInt | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ceiling_h: typing.SupportsFloat | typing.SupportsIndex, flood_eps: typing.SupportsFloat | typing.SupportsIndex, ratio_cap: typing.SupportsFloat | typing.SupportsIndex, boil_rate: typing.SupportsFloat | typing.SupportsIndex, boil_p_thresh: typing.SupportsFloat | typing.SupportsIndex, steam_yield: typing.SupportsFloat | typing.SupportsIndex, gas_energy: typing.Any = None, gas_conservative: typing.Any = None, thermal_solid: typing.Any = None, is_vacuum: typing.Any = None, is_ambient: typing.Any = None, t_amb_raw: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def step_water_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], steam_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ceiling_h: typing.SupportsFloat | typing.SupportsIndex, flood_eps: typing.SupportsFloat | typing.SupportsIndex, ratio_cap: typing.SupportsFloat | typing.SupportsIndex, boil_rate: typing.SupportsFloat | typing.SupportsIndex, boil_p_thresh: typing.SupportsFloat | typing.SupportsIndex, steam_yield: typing.SupportsFloat | typing.SupportsIndex, ambient: bool, t_amb_raw: typing.SupportsInt | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, wdt: typing.SupportsFloat | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex, active: typing.Any = None, n_env: typing.SupportsInt | typing.SupportsIndex = 1, engines: typing.Any = None) -> None:
        ...
    def step_water_tail(self, water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], before: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], steam_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ceiling_h: typing.SupportsFloat | typing.SupportsIndex, flood_eps: typing.SupportsFloat | typing.SupportsIndex, ratio_cap: typing.SupportsFloat | typing.SupportsIndex, boil_rate: typing.SupportsFloat | typing.SupportsIndex, boil_p_thresh: typing.SupportsFloat | typing.SupportsIndex, steam_yield: typing.SupportsFloat | typing.SupportsIndex, gas_energy: typing.Any = None, gas_conservative: typing.Any = None, thermal_solid: typing.Any = None, is_vacuum: typing.Any = None, is_ambient: typing.Any = None, t_amb_raw: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def sync_resident_fleck_plane(self, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        #29 P4a: copy the last resident sweep's Fleck plane into engine.radiation (on demand; never per tick).
        """
    def upload_light_resident(self, dev: dict, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, gas_light_absorb_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_light_glow_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], light_cones: typing.Any = None, light_sky: typing.Any = None, light_emit_mult: typing.Any = None) -> None:
        ...
    def water_substep_count(self, sim_time: typing.SupportsFloat | typing.SupportsIndex) -> int:
        ...
    @property
    def atmos(self) -> AtmosphereSolver:
        ...
    @property
    def combustion(self) -> CombustionSolver:
        ...
    @property
    def e_water_evac_export_sum(self) -> int:
        ...
    @property
    def edit_queue_books(self) -> list[int]:
        """
        #29 P4c: the device edit queue's books of the last tick it ran (EDIT_QUEUE_LAYOUT slots).
        """
    @property
    def emissive(self) -> EmissiveTable:
        ...
    @property
    def eos(self) -> EOSSolver:
        ...
    @property
    def fire(self) -> FireSimulation:
        ...
    @property
    def headless(self) -> bool:
        """
        #29 P4a: skip the render-only W6a ripple (headless runs).
        """
    @headless.setter
    def headless(self, arg0: bool) -> None:
        ...
    @property
    def light_emission(self) -> LightEmissionTable:
        ...
    @property
    def light_requested(self) -> bool:
        """
        P6a: when True, step_tail's sweep carries the light channels into the light planes it is handed (all five then required); False (the default) computes no light and leaves those planes untouched.
        """
    @light_requested.setter
    def light_requested(self, arg0: bool) -> None:
        ...
    @property
    def light_transport(self) -> int:
        """
        P6d: the light group's transport at the one sweep invocation (run_sweep_, both step_tail and relight(), both backends) -- RadiationSweep.STEP (default) or .SHEAR. Bound once from [light] transport; heat's transport is unaffected.
        """
    @light_transport.setter
    def light_transport(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def radiation(self) -> RadiationSweep:
        ...
    @property
    def temperature(self) -> TemperatureSolver:
        ...
    @property
    def water(self) -> WaterSolver:
        ...
class RadiationSweep:
    CONE_TURN: typing.ClassVar[int] = 65536
    F_ONE: typing.ClassVar[int] = 16777216
    F_SHIFT: typing.ClassVar[int] = 24
    HEAT_ABSORB_Q_MAX: typing.ClassVar[int] = 268435456
    LIGHT_CONE_BUDGET: typing.ClassVar[int] = 17592186044416
    LIGHT_CONE_ROW_MAX: typing.ClassVar[int] = 281474976710656
    LIGHT_EMIT_MULT_MAX: typing.ClassVar[int] = 1048576
    LIGHT_SKY_MAX: typing.ClassVar[int] = 1099511627776
    N_GAS_PLANES_MAX: typing.ClassVar[int] = 16
    SHEAR: typing.ClassVar[int] = 1
    STEP: typing.ClassVar[int] = 0
    @staticmethod
    def cone_emission(rgb: typing.Annotated[collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], "FixedSize(3)"], center_q: typing.SupportsInt | typing.SupportsIndex, spread_q: typing.SupportsInt | typing.SupportsIndex, n_ordinates: typing.SupportsInt | typing.SupportsIndex = 16) -> numpy.typing.NDArray[numpy.int64]:
        """
        P6b: one cone's per-ordinate emission (n_ordinates, 3) int64 -- the power projected onto the ordinates' angular bins by overlap (sweep_ref_q.cone_emission). Angles in Q16 turns.
        """
    @staticmethod
    def fleck_f_gas_q24(e_table: EmissiveTable, T_q: typing.SupportsInt | typing.SupportsIndex, a_gas_q: typing.SupportsInt | typing.SupportsIndex, n_bulk: typing.SupportsInt | typing.SupportsIndex, n_floor_q: typing.SupportsInt | typing.SupportsIndex, recip_cv: typing.SupportsInt | typing.SupportsIndex, t_amb_q: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        The GAS-arm Fleck factor (Q24) for one gas cell whose smoke term absorbs a_gas_q, exactly as the sweep's pre-pass forms it (P5b): L priced in the temperature fold's gas currency (n_floor_q, recip_cv — PhysicsEngine.gas_capacity_q()).
        """
    @staticmethod
    def fleck_f_solid_q24(e_table: EmissiveTable, T_q: typing.SupportsInt | typing.SupportsIndex, a_q: typing.SupportsInt | typing.SupportsIndex, his: typing.SupportsInt | typing.SupportsIndex, t_amb_q: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        The solid-branch Fleck factor (Q24) for one cell, exactly as the sweep's pre-pass forms it — the tile inspector's `f` row.
        """
    @staticmethod
    def gas_extinction_q16(densities: numpy.typing.NDArray[numpy.int32], heat_absorb_q16: numpy.typing.NDArray[numpy.int32], n_bulk: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        a_gas (Q16) for ONE gas cell: min(ONE, sum_g heat_absorb_q16[g] * max(0, N_g) >> 16), 0 below the N_EPS bulk floor — the smoke term the sweep's pre-pass forms (P5a), for the tile inspector (P5c).
        """
    @staticmethod
    def ordinate_constants(n_ordinates: typing.SupportsInt | typing.SupportsIndex, transport: typing.SupportsInt | typing.SupportsIndex) -> list:
        """
        The checked-in per-ordinate constants as (sx, sy, x_major, s_m) tuples, for the recompute test.
        """
    @staticmethod
    def ordinate_dirs(n_ordinates: typing.SupportsInt | typing.SupportsIndex) -> list:
        """
        P6a: the checked-in per-ordinate direction cosines as (mu_q, eta_q) Q16 tuples, for the recompute test.
        """
    def __init__(self) -> None:
        ...
    def a_eff_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        P5a: a COPY of the extinction plane (Q16, (h, w)) the last run() READ — the material extinction with the smoke term on gas cells.
        """
    def d_eff_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        P5a: a COPY of the stamped-total plane (Q16, (h, w)) the last run() READ — max(dyn_heat_atten_q, the effective extinction).
        """
    def derive_ambient(self, is_vacuum: numpy.typing.NDArray[numpy.bool], e_table: EmissiveTable, vac_level: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
        """
        A COPY of the per-cell ambient LEVEL plane derived from vacuum/interior state (thermal model v2 R3): a vacuum cell takes vac_level, every other cell E°[0]. vac_level < 0 means E°[0] (R4, the shipped uniform answer); above E°[0] raises.
        """
    def fleck_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        A COPY of the Fleck plane (Q24, (h, w)) the last run() computed.
        """
    def light_a_eff_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        P6a: a COPY of the light material share a_c (Q16, (h, w, 3)) the last CPU light run READ -- the smoke term folded in on gas cells.
        """
    def light_d_eff_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        P6a: a COPY of the stamped light total d_c (Q16, (h, w, 3)) the last CPU light run READ.
        """
    def light_gcoef_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        P6a: a COPY of the glow coefficient g_c (Q16, (h, w, 3)) of the last CPU light run -- 0 off the gas cells.
        """
    def run(self, temperature: numpy.typing.NDArray[numpy.int32], heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], heat_inv_shift: numpy.typing.NDArray[numpy.int32], thermal_solid: numpy.typing.NDArray[numpy.bool], e_table: EmissiveTable, amb_level: typing.Any, t_amb_q: typing.SupportsInt | typing.SupportsIndex, k_leak_q: typing.SupportsInt | typing.SupportsIndex, transport: typing.SupportsInt | typing.SupportsIndex, n_ordinates: typing.SupportsInt | typing.SupportsIndex, rad_net: numpy.typing.NDArray[numpy.int64], rad_flux: numpy.typing.NDArray[numpy.int64], rad_amb: numpy.typing.NDArray[numpy.int64], rad_fluence: numpy.typing.NDArray[numpy.int64], fleck_enabled: bool = True, gas: typing.Any = None, heat_absorb_q16: typing.Any = None, n_bulk: typing.Any = None, n_floor_q: typing.SupportsInt | typing.SupportsIndex = 0, recip_cv: typing.SupportsInt | typing.SupportsIndex = 0, light_atten_q: typing.Any = None, dyn_light_atten_q: typing.Any = None, light_q: typing.Any = None, light_flux_q: typing.Any = None, light_glow: typing.Any = None, light_absorb_q16: typing.Any = None, light_glow_q16: typing.Any = None, light_table: typing.Any = None, light_transport: typing.SupportsInt | typing.SupportsIndex = 0, light_cones: typing.Any = None, light_sky: typing.Any = None, light_emit_mult: typing.Any = None) -> None:
        """
        One tick of the sweep over all ordinates. It OVERWRITES the four int64 planes — zeroed here before the first ordinate, so they hold the last run's values until the next run and the tile inspector can read them at render time (design row 38); no caller wipe needed. transport: RadiationSweep.STEP or .SHEAR; n_ordinates: 16 or 12. fleck_enabled=False is the reference's undamped (f_plane=None) configuration, for the gates. amb_level is the PER-CELL ambient LEVEL plane (int64 (h, w), 0 <= amb <= e_table[0]) the sweep radiates against — None means E°[0] everywhere, the uniform thermal-v2-R4 configuration. gas (int32 (n_gases, h, w)), heat_absorb_q16 (int32 (n_gases,)) and n_bulk (int32 (h, w)) — all or none — are the SMOKE TERM (P5a, design v3 §6.3): a gas cell reads a = max(a, min(ONE, Σ hq·N >> 16)), 0 below the N_EPS bulk floor, and d = max(d, a). None is the pre-P5a sweep. With the group, n_floor_q and recip_cv (both > 0) are the temperature fold's gas currency (PhysicsEngine.gas_capacity_q()), in which the Fleck pre-pass's GAS ARM (P5b, design v3 §2.8) damps every absorbing gas cell: L = deposit_dT_wide_i64((a_gas·ex) >> 16, reciprocal_q16(max(N, n_floor_q)), recip_cv). P6a THE LIGHT GROUP (all five planes or none): light_atten_q / dyn_light_atten_q int32 (h, w, 3), light_q / light_glow int64 (h, w, 3) and light_flux_q int64 (h, w, 2), OVERWRITTEN; light_absorb_q16 / light_glow_q16 int32 (n_gases, 3) the smoke term (with the gas group); light_table int64 (3, E_TABLE_SIZE) or None = the checked-in L°; light_transport STEP (default) or SHEAR. The books land in RadiationSweep.light_books. P6b: light_cones int64 (n, 7) rows (y, x, r, g, b, center_q, spread_q) -- CONE EMITTERS, the power projected onto the ordinates' bins by overlap and entering the cell's stream -- and light_sky int64 (n_ordinates, 3), the virtual ring's per-ordinate outflow (None: the dark ring).
        """
    @property
    def light_books(self) -> dict:
        """
        P6a: {'emit', 'absorb', 'ring_in', 'ring_out': (R, G, B), 'min_stream', 'max_stream'} of the last light-carrying run -- emit + ring_in == absorb + ring_out per channel, exactly.
        """
    @property
    def max_stream(self) -> int:
        ...
    @property
    def min_stream(self) -> int:
        ...
class TemperatureSolver:
    def __init__(self) -> None:
        ...
    def book_solid_seam(self, delta: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        #73 item 2: book a topology seam's MEASURED change to the solid books (int64, cap_real*T units) -- adds it to e_solid_seam_sum AND to the solid_energy_books_sum snapshot, so the snapshot stays the live sum when the seam fires after this tick's step(). Called by GameMap._solid_energy_book only.
        """
    def step(self, temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat_inv_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], face_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Any = None, wind_y: typing.Any = None, dt: typing.SupportsFloat | typing.SupportsIndex = 0.0, n_bulk: typing.Any = None, thermal_solid: typing.Any = None, rad_net: typing.Any = None, rad_fluence: typing.Any = None, e_table: typing.Any = None, clamp_enabled: bool = True, gas_energy: typing.Any = None, t_amb_q: typing.SupportsInt | typing.SupportsIndex = 0, is_ambient: typing.Any = None, rad_fine_bits: typing.SupportsInt | typing.SupportsIndex = -1) -> None:
        ...
    @property
    def T_MAX_PHYS(self) -> float:
        ...
    @T_MAX_PHYS.setter
    def T_MAX_PHYS(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def c_v(self) -> float:
        ...
    @c_v.setter
    def c_v(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def cond_limit_hits(self) -> int:
        ...
    @property
    def dbg_T_post_conduction(self) -> int:
        ...
    @property
    def dbg_T_post_cooling(self) -> int:
        ...
    @property
    def dbg_T_post_heat(self) -> int:
        ...
    @property
    def dbg_probe_idx(self) -> int:
        ...
    @dbg_probe_idx.setter
    def dbg_probe_idx(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def e_cond_cap_sum(self) -> int:
        ...
    @property
    def e_cond_trunc_sum(self) -> int:
        ...
    @property
    def e_dep_floor_drop_sum(self) -> int:
        ...
    @property
    def e_deposit_drop_sum(self) -> int:
        ...
    @property
    def e_gas_cond_sum(self) -> int:
        ...
    @property
    def e_gas_deposit_sum(self) -> int:
        ...
    @property
    def e_gas_rail_sum(self) -> int:
        ...
    @property
    def e_rad_boundary_export_sum(self) -> int:
        ...
    @property
    def e_rad_clamp_drop_sum(self) -> int:
        ...
    @property
    def e_rad_floor_drop_sum(self) -> int:
        ...
    @property
    def e_ring_pin_sum(self) -> int:
        ...
    @property
    def e_solid_cond_sum(self) -> int:
        ...
    @property
    def e_solid_deposit_sum(self) -> int:
        ...
    @property
    def e_solid_low_rail_sum(self) -> int:
        ...
    @property
    def e_solid_seam_sum(self) -> int:
        ...
    @property
    def e_vac_wipe_sum(self) -> int:
        ...
    @property
    def gas_advection_rate(self) -> float:
        ...
    @gas_advection_rate.setter
    def gas_advection_rate(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def n_floor_heat(self) -> float:
        ...
    @n_floor_heat.setter
    def n_floor_heat(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def no_face(self) -> int:
        ...
    @no_face.setter
    def no_face(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def o2_vacuum_thresh(self) -> float:
        ...
    @o2_vacuum_thresh.setter
    def o2_vacuum_thresh(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rad_clamp_hits(self) -> int:
        ...
    @property
    def solid_energy_books_sum(self) -> int:
        ...
    @property
    def t_low_rail_hits(self) -> int:
        ...
    @property
    def t_max_phys_hits(self) -> int:
        ...
class UnitEventRingBuffer:
    """
    #29 P6.3: the per-env device event ring (N, E_max, F) int64 (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, n_units: typing.SupportsInt | typing.SupportsIndex, e_max: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def begin(self) -> None:
        """
        Empty every env's ring (tick stream).
        """
    def download(self) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
        """
        (records (N, E_max, F), counts (N,), dropped (N,)) int64.
        """
    @property
    def shape(self) -> tuple[int, int, int]:
        ...
class UnitRecordListBuffer:
    """
    #29 P6.4a: a per-env device record list (N, cap, width) int64 with per-row staging (Class T): slot 1b's request list.
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, n_units: typing.SupportsInt | typing.SupportsIndex, cap: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, row_cap: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def begin(self) -> None:
        """
        Empty every env's list (tick stream).
        """
    def counts(self) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
        """
        #29 P7a.3: (counts (N,), dropped (N,)) int64 alone (no record copy).
        """
    def download(self) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
        """
        (records (N, cap, width), counts (N,), dropped (N,)) int64.
        """
    def upload(self, recs: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], counts: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        """
        #29 P7a.3: set every env's list and count; dropped and staging to 0.
        """
    @property
    def shape(self) -> tuple[int, int, int]:
        ...
class UnitStoreBuffer:
    """
    #29 P6.1: the (N, U, W) int64 device unit store (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, n_units: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def download(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        The device store as a fresh (N, U, W) int64 array.
        """
    def upload(self, rows: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        """
        Copy (N, U, W) int64 rows into the device store (tick stream).
        """
    @property
    def ptr(self) -> int:
        ...
    @property
    def shape(self) -> tuple[int, int, int]:
        ...
class UnitTablesBuffer:
    """
    #29 P6.2: the (N, T) int64 device tables of the unit kernels (Class T).
    """
    def __init__(self, n_env: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def download(self) -> numpy.typing.NDArray[numpy.int64]:
        ...
    def upload(self, rows: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
        ...
    @property
    def shape(self) -> tuple[int, int]:
        ...
class WaterSolver:
    def __init__(self) -> None:
        ...
    def max_dt(self) -> float:
        ...
    def max_dt_q(self) -> int:
        ...
    def ripple_max_dt(self) -> float:
        ...
    def step(self, water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vx: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vy: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], floor_height: typing.Any = None, atmosphere: typing.Any = None, solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def step_ripple(self, ripple: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], ripple_v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Any = None, p_prev: typing.Any = None, solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def damping(self) -> float:
        ...
    @damping.setter
    def damping(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def depth_eps(self) -> float:
        ...
    @depth_eps.setter
    def depth_eps(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def dx(self) -> float:
        ...
    @dx.setter
    def dx(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def g(self) -> float:
        ...
    @g.setter
    def g(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def gamma_r(self) -> float:
        ...
    @gamma_r.setter
    def gamma_r(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def h_cap(self) -> float:
        ...
    @h_cap.setter
    def h_cap(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def h_ref(self) -> float:
        ...
    @h_ref.setter
    def h_ref(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_amp(self) -> float:
        ...
    @k_amp.setter
    def k_amp(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_p(self) -> float:
        ...
    @k_p.setter
    def k_p(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def k_splash(self) -> float:
        ...
    @k_splash.setter
    def k_splash(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def v_max(self) -> float:
        ...
    @v_max.setter
    def v_max(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
def applier_blast_probe(applier: FieldApplierBuffer, tables: UnitTablesBuffer, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, q: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], M: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P7c.2 TEST-ONLY: the blast walk and air read per query (N, K, 3 + 2 M).
    """
def applier_layout() -> dict:
    """
    #29 P6.4b: the compiled applier layout (applier_layout.h).
    """
def applier_run(applier: FieldApplierBuffer, store: UnitStoreBuffer, tables: UnitTablesBuffer, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, stage: typing.SupportsInt | typing.SupportsIndex, ring: typing.Any = None, cells: typing.Any = None, door: typing.Any = None, claims: typing.Any = None, noise: typing.Any = None, blast_log: typing.Any = None, door_table: typing.Any = None, door_tiles: typing.Any = None, duct_table: typing.Any = None, vent_table: typing.Any = None, source_table: typing.Any = None, source_tiles: typing.Any = None, source_out: typing.SupportsInt | typing.SupportsIndex = 0, signal_tables: typing.Any = None, tick_row: typing.SupportsInt | typing.SupportsIndex = 0, timed_table: typing.Any = None, heat_table: typing.Any = None, horde: typing.Any = None) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P6.4b / P7a.1: the per-env serial applier over the effect lists in a stage (applier_layout.h's APPLIER_STAGE_*); returns (books (N, N_BOOKS), coverage (N,): the rows the walk changed, ring_dropped (N,)) int64.
    """
def atan2_q16(y: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Q2-LIFT: pure-integer atan2 on Q16.16 (radians out, Q16.16; range [-205887, +205887] == [-quantize(pi), +quantize(pi)]).
    """
def build_id() -> str:
    """
    git HEAD's SHA this module was built from (+ '-dirty' when a tracked file differed; 'unknown' without git). Generated at build time (cpp/gen_build_id.cmake).
    """
def bulk_flux_transport(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    EOS P1: donor-cell conservative flux transport of every `gas_conservative`-flagged plane, once, on the given wind field.
    """
def conduction_cell_capacity_q(is_ts: bool, heat_inv_shift: typing.SupportsInt | typing.SupportsIndex, n_raw: typing.SupportsInt | typing.SupportsIndex, n_floor_q: typing.SupportsInt | typing.SupportsIndex, c_v_q: typing.SupportsInt | typing.SupportsIndex) -> tuple[int, int]:
    """
    temperature_solver.h conduction::cell_capacity_q -> (cap_used, cap_real). THE one capacity law both backends call.
    """
def conductor_books_begin(cap: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    #29 P7e.4: open the deferral (every unit-side count copied D2D into one arena).
    """
def conductor_books_deferring() -> bool:
    ...
def conductor_books_end() -> tuple[numpy.typing.NDArray[numpy.int64], list]:
    """
    #29 P7e.4: THE ONE READ: (words, [(label, kind, offset, n)]).
    """
def conductor_label(label: str) -> None:
    """
    #29 P7e.4: the conductor's current call line.
    """
def cos_q16(a: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Q2-LIFT: pure-integer cos on Q16.16 radians (output Q16.16 in [-65536, 65536]; accuracy pinned for |a| <= 4*pi, any int32 defined).
    """
def cuda_available() -> bool:
    """
    True if a CUDA device is present and usable.
    """
def cuda_bulk_flux_transport(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    P6.1 isolated: GPU donor-cell conservative flux transport of every `gas_conservative`-flagged plane, once, on the given wind field (bit-identical to bulk_flux_transport).
    """
def cuda_combustion_step(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], o2_idx: typing.SupportsInt | typing.SupportsIndex, inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, black_smoke_idx: typing.SupportsInt | typing.SupportsIndex, temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], ignition_temp_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dt: typing.SupportsFloat | typing.SupportsIndex, c_v: typing.SupportsFloat | typing.SupportsIndex, n_floor_heat: typing.SupportsFloat | typing.SupportsIndex, burn_rate: typing.SupportsFloat | typing.SupportsIndex, o2_thresh_burn: typing.SupportsFloat | typing.SupportsIndex, H_FUEL_M: typing.SupportsFloat | typing.SupportsIndex, H_FUEL_SHIFT: typing.SupportsInt | typing.SupportsIndex, soot_yield: typing.SupportsFloat | typing.SupportsIndex, fuel_per_o2: typing.SupportsFloat | typing.SupportsIndex, o2_frac_ext: typing.SupportsFloat | typing.SupportsIndex, o2_frac_full: typing.SupportsFloat | typing.SupportsIndex, T_MAX_PHYS: typing.SupportsFloat | typing.SupportsIndex, thermal_solid: typing.Any = None, heat_inv_shift: typing.Any = None, heat: typing.Any = None, H_BED_M: typing.SupportsFloat | typing.SupportsIndex = 0.0, H_BED_SHIFT: typing.SupportsInt | typing.SupportsIndex = 0, dem_acc: typing.Any = None, draw_r: typing.SupportsInt | typing.SupportsIndex = 1, dyn_permeability: typing.Any = None, max_claimants: typing.SupportsInt | typing.SupportsIndex = 4, fire_T_ext: typing.SupportsFloat | typing.SupportsIndex = 350.0, fire_T_span: typing.SupportsFloat | typing.SupportsIndex = 180.0, hotf_cap: typing.SupportsFloat | typing.SupportsIndex = 10.0, fire_T_ext_plane: typing.Any = None, fuel_per_o2_plane: typing.Any = None, atmosphere: typing.Any = None, p_ext_q: typing.SupportsInt | typing.SupportsIndex = 0, p_full_q: typing.SupportsInt | typing.SupportsIndex = 0, eos: typing.Any = None, gas_energy: typing.Any = None, is_ambient: typing.Any = None, t_amb_q: typing.SupportsInt | typing.SupportsIndex = 0, combustion: typing.Any = None) -> tuple:
    """
    P6.9b isolated: run ONE GPU combustion step (the two-gather reformulation, continuous-O2 proportional demand) in place on the three gas planes + temperature + wall_hp (bit-identical to CombustionSolver.step) and return the (heat_floor_hits, t_max_phys_hits, e_deposit_drop_sum) per-call rail counts.
    """
def cuda_device_info() -> str:
    """
    GPU name + compute capability + runtime/driver versions.
    """
def cuda_eos_energy_flux(gas_energy: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, adiabatic_index: typing.SupportsFloat | typing.SupportsIndex, t_amb_k: typing.SupportsFloat | typing.SupportsIndex, c_value: typing.SupportsFloat | typing.SupportsIndex, t_min: typing.SupportsFloat | typing.SupportsIndex, t_max_phys: typing.SupportsFloat | typing.SupportsIndex, is_ambient: typing.Any = None, thermal_solid: typing.Any = None) -> tuple:
    """
    P-G2 isolated: run the GPU face-flux energy step (K3, sub-cycled n_sub times) + the once-per-tick recovery in place on gas_energy/temperature. `atmosphere` MUST be the ABSOLUTE solved pressure (post step-5 un-shift on an ambient map). Returns (counters[FLUX_CNT_SLOTS],) — see cuda_kick_compression.h for the slot map.
    """
def cuda_eos_kick_compression(wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_energy: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], p_new: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, cap2_plane: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], c_max: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, adiabatic_index: typing.SupportsFloat | typing.SupportsIndex, absorb_strength: typing.SupportsFloat | typing.SupportsIndex, n_floor_solver: typing.SupportsFloat | typing.SupportsIndex, u_max: typing.SupportsFloat | typing.SupportsIndex, k_drag: typing.SupportsFloat | typing.SupportsIndex = 0.0, k_drag2: typing.SupportsFloat | typing.SupportsIndex = 0.0, t_amb_k: typing.SupportsFloat | typing.SupportsIndex = 290.0, thermal_solid: typing.Any = None) -> tuple:
    """
    P-G2 isolated: run the GPU kick (K1 only — K2 the old step-4c compression-work kernel is DELETED, arc #54) in place on wind_x/wind_y/gas_energy; cap2_plane is the per-cell (h,w) int64 velocity-cap-squared plane (Q32.32 raw) — MUST be >= 0 everywhere. Returns (digest_velocity, counters[KICK_CNT_SLOTS]) for this call — see cuda_kick_compression.h for the slot map.
    """
def cuda_eos_mg_solve(solver: EOSSolver, pstar: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], div_u: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, p_out: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> tuple:
    """
    P6.3 isolated: run the multigrid pressure solve with the hierarchy built host-side (the SAME mg_build_levels the CPU calls) and the ENTIRE V-cycle iteration on the GPU; writes the solved P into p_out and returns (digest, launches_actual, launches_naive) — the digest is bit-identical to eos_mg_solve_ref / digest_helmholtz.
    """
def cuda_eos_sl_advect(wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, thermal_solid: typing.Any = None) -> int:
    """
    P6.2 isolated: run the GPU fused SL-advection substep chain in place on wind_x/wind_y/temperature (bit-identical to eos_sl_advect_ref) and return the chained FNV digest. dx (tile size, m) is REQUIRED: the displacement is eos_transport_step_q(dt, n_sub, dx) tiles per m/s (#217 patch U).
    """
def cuda_fire_step(fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_o2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, k_grow: typing.SupportsFloat | typing.SupportsIndex, k_die: typing.SupportsFloat | typing.SupportsIndex, fire_T_ext: typing.SupportsFloat | typing.SupportsIndex, fire_T_span: typing.SupportsFloat | typing.SupportsIndex, fuel_ref: typing.SupportsFloat | typing.SupportsIndex, o2_frac_ext: typing.SupportsFloat | typing.SupportsIndex, o2_frac_full: typing.SupportsFloat | typing.SupportsIndex, I_min: typing.SupportsFloat | typing.SupportsIndex, k_wind_fan: typing.SupportsFloat | typing.SupportsIndex, k_wind_strip: typing.SupportsFloat | typing.SupportsIndex, wall_damage: typing.SupportsFloat | typing.SupportsIndex, temp_scale: typing.SupportsFloat | typing.SupportsIndex, I_cap_per_avail: typing.SupportsFloat | typing.SupportsIndex = 2.5299999713897705, o2_frac_amb: typing.SupportsFloat | typing.SupportsIndex = 0.20999999344348907, o2f_cap: typing.SupportsFloat | typing.SupportsIndex = 5.0, hotf_cap: typing.SupportsFloat | typing.SupportsIndex = 10.0, fuel_recip: typing.Any = None, fire_T_ext_plane: typing.Any = None, p_ext_q: typing.SupportsInt | typing.SupportsIndex = 0, p_full_q: typing.SupportsInt | typing.SupportsIndex = 0) -> list:
    """
    P6.8 isolated: run ONE GPU fire step (re-derived — continuous-O2 mole-fraction gate) in place on fire/smoke/wall_hp (bit-identical to FireSimulation.step) and return the destroyed-walls list of (y,x) tuples. temperature is still a parameter but is READ ONLY as of P-R2 (the plume->T shim write is deleted).
    """
def cuda_map_mul_q16(in_: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], factor_q16: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
    """
    S0 hello-world: out[i] = mul_q16(in[i], factor_q16) computed on the GPU via the shared toolkit; bit-identical to the CPU mul_q16.
    """
def cuda_radiation_sweep_launch_count(transport: typing.SupportsInt | typing.SupportsIndex, n_ordinates: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, light: bool = False) -> int:
    """
    Kernel launches one sweep of this shape issues (3 bookkeeping + one per wavefront index; design v3 section 10; with light (P6a) the h + w - 1 anti-diagonals + one glow launch). -1 if unsupported.
    """
def cuda_radiation_sweep_resident(n_env: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, d_temperature: typing.SupportsInt | typing.SupportsIndex, d_heat_atten_q: typing.SupportsInt | typing.SupportsIndex, d_dyn_heat_atten_q: typing.SupportsInt | typing.SupportsIndex, d_heat_inv_shift: typing.SupportsInt | typing.SupportsIndex, d_thermal_solid: typing.SupportsInt | typing.SupportsIndex, d_amb_level: typing.SupportsInt | typing.SupportsIndex, d_is_vacuum: typing.SupportsInt | typing.SupportsIndex, d_e_table: typing.SupportsInt | typing.SupportsIndex, e_fine_bits: typing.SupportsInt | typing.SupportsIndex, d_vac_level: typing.SupportsInt | typing.SupportsIndex, d_k_leak_q: typing.SupportsInt | typing.SupportsIndex, d_t_amb_q: typing.SupportsInt | typing.SupportsIndex, d_gas: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, d_heat_absorb_q16: typing.SupportsInt | typing.SupportsIndex, d_n_bulk: typing.SupportsInt | typing.SupportsIndex, n_floor_q: typing.SupportsInt | typing.SupportsIndex, recip_cv: typing.SupportsInt | typing.SupportsIndex, transport: typing.SupportsInt | typing.SupportsIndex, n_ordinates: typing.SupportsInt | typing.SupportsIndex, fleck_enabled: bool, d_outflow: typing.SupportsInt | typing.SupportsIndex, d_amb_m: typing.SupportsInt | typing.SupportsIndex, d_ex_cell: typing.SupportsInt | typing.SupportsIndex, d_f_q24: typing.SupportsInt | typing.SupportsIndex, d_a_eff: typing.SupportsInt | typing.SupportsIndex, d_d_eff: typing.SupportsInt | typing.SupportsIndex, d_rad_net: typing.SupportsInt | typing.SupportsIndex, d_rad_flux: typing.SupportsInt | typing.SupportsIndex, d_rad_amb: typing.SupportsInt | typing.SupportsIndex, d_rad_fluence: typing.SupportsInt | typing.SupportsIndex, d_cnt: typing.SupportsInt | typing.SupportsIndex, light: typing.Any = None) -> int:
    """
    P4 (TEST/BENCH): the sweep's LAUNCH CORE on raw device pointers (CuPy .data.ptr uintptr_t; 0 == nullptr for d_amb_level/d_is_vacuum), (N, h, w)-shaped. #78: e_fine_bits is the device table's currency (the EmissiveTable.fine_bits it was baked with; REQUIRED -- a table on the device carries none of its own). P5a: d_gas (N, n_gases, h, w) / d_heat_absorb_q16 (n_gases,) / d_n_bulk (N, h, w) are the smoke term, all three or all 0 (with n_gases 0); P5b: n_floor_q / recip_cv are the gas arm's currency, two host scalars, both > 0 with the smoke term (0 without); d_a_eff / d_d_eff are required (N, h, w) int32 scratch. Launch only: no malloc, no transfer, no sync — the caller synchronizes and reads the (N, RADIATION_SWEEP_CNT_SLOTS) counter block. Returns the launch count. P6a: `light` is None or a dict of the light group's device pointers by RadiationLightDev field name (d_light_atten_q ... d_light_glow, (N, ...)-shaped; the table and the two smoke columns shared) plus 'transport'.
    """
def cuda_radiation_sweep_run(temperature: numpy.typing.NDArray[numpy.int32], heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], heat_inv_shift: numpy.typing.NDArray[numpy.int32], thermal_solid: numpy.typing.NDArray[numpy.bool], e_table: EmissiveTable, amb_level: typing.Any, t_amb_q: typing.SupportsInt | typing.SupportsIndex, k_leak_q: typing.SupportsInt | typing.SupportsIndex, transport: typing.SupportsInt | typing.SupportsIndex, n_ordinates: typing.SupportsInt | typing.SupportsIndex, rad_net: numpy.typing.NDArray[numpy.int64], rad_flux: numpy.typing.NDArray[numpy.int64], rad_amb: numpy.typing.NDArray[numpy.int64], rad_fluence: numpy.typing.NDArray[numpy.int64], fleck_enabled: bool = True, is_vacuum: typing.Any = None, vac_level: typing.SupportsInt | typing.SupportsIndex = -1, fleck_out: typing.Any = None, gas: typing.Any = None, heat_absorb_q16: typing.Any = None, n_bulk: typing.Any = None, n_floor_q: typing.SupportsInt | typing.SupportsIndex = 0, recip_cv: typing.SupportsInt | typing.SupportsIndex = 0, light_atten_q: typing.Any = None, dyn_light_atten_q: typing.Any = None, light_q: typing.Any = None, light_flux_q: typing.Any = None, light_glow: typing.Any = None, light_absorb_q16: typing.Any = None, light_glow_q16: typing.Any = None, light_table: typing.Any = None, light_transport: typing.SupportsInt | typing.SupportsIndex = 0, light_cones: typing.Any = None, light_sky: typing.Any = None, light_emit_mult: typing.Any = None) -> tuple:
    """
    P4 isolated: ONE radiation sweep on the GPU (the per-call path PhysicsEngine.step_tail dispatches), bit-identical to RadiationSweep.run on the same arguments. amb_level None + is_vacuum None is the uniform E0 door; amb_level None + is_vacuum given DERIVES the ambient on the device (derive_ambient's twin). gas / heat_absorb_q16 / n_bulk (int32, all or none) are the smoke term (P5a); only the gases with a non-zero coefficient cross the bus; with them n_floor_q / recip_cv (both > 0) are the gas arm's currency (P5b). Overwrites the four int64 planes (untouched on an ingress rejection, which raises ValueError). Returns (min_stream, max_stream, launches).
    """
def cuda_spike_add1(dev_ptr: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    S8a spike: int32 in-place +1 on a raw device pointer (CuPy .data.ptr).
    """
def cuda_swarm_resident_probe(species: str, columns: collections.abc.Sequence, high_water: typing.Any, temperature: typing.Any, solid: typing.Any, env: typing.Any, tables: typing.Any, death_rec: typing.Any) -> int:
    """
    arc #63 P3 TEST-ONLY (v4.7 G-b): the RESIDENT launch core on device copies of the caller's arrays, padding included, copied back whole.
    """
def cuda_swarm_species_step(species: str, columns: collections.abc.Sequence, high_water: typing.Any, temperature: typing.Any, solid: typing.Any, env: typing.Any, tables: typing.Any, death_rec: typing.Any) -> int:
    """
    arc #63 P3: swarm_species_step on the per-call CUDA wrapper.
    """
def cuda_temperature_step(temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat_inv_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], face_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], no_face: typing.SupportsInt | typing.SupportsIndex, o2_vacuum_thresh: typing.SupportsFloat | typing.SupportsIndex, c_v: typing.SupportsFloat | typing.SupportsIndex, n_floor_heat: typing.SupportsFloat | typing.SupportsIndex, gas_advection_rate: typing.SupportsFloat | typing.SupportsIndex, t_max_phys: typing.SupportsFloat | typing.SupportsIndex, n_bulk: typing.Any = None, wind_x: typing.Any = None, wind_y: typing.Any = None, dt: typing.SupportsFloat | typing.SupportsIndex = 0.0, thermal_solid: typing.Any = None, gas_energy: typing.Any = None, t_amb_k: typing.SupportsFloat | typing.SupportsIndex = 290.0, rad_net: typing.Any = None, rad_fluence: typing.Any = None, e_table: typing.Any = None, is_ambient: typing.Any = None, rad_fine_bits: typing.SupportsInt | typing.SupportsIndex = -1) -> tuple:
    """
    P6.6/P-G2 isolated: run the GPU unified temperature solver in place on `temperature` (+ `gas_energy` when supplied — bit-identical to TemperatureSolver.step); returns (t_max_phys_hits, e_cond_trunc_sum, e_cond_cap_sum, cond_limit_hits, e_vac_wipe_sum, e_ring_pin_sum, e_deposit_drop_sum, e_gas_deposit_sum, e_gas_cond_sum, e_gas_rail_sum, e_solid_deposit_sum, e_solid_cond_sum, rad_clamp_hits, solid_energy_books_sum, e_rad_clamp_drop_sum, e_rad_boundary_export_sum, e_rad_floor_drop_sum, e_solid_low_rail_sum, e_dep_floor_drop_sum) for this call (P-E2a + P-E2b + arc #54 + P-G5 + P5c + #73; solid_energy_books_sum is a SNAPSHOT, not a per-call delta).
    """
def cuda_water_step(water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vx: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vy: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], floor_height: typing.Any = None, atmosphere: typing.Any = None, solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, damping: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, k_p: typing.SupportsFloat | typing.SupportsIndex, v_max: typing.SupportsFloat | typing.SupportsIndex, depth_eps: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    S3 isolated: run the GPU water solver in place on water_depth/flow_vx/flow_vy (bit-identical to WaterSolver.step).
    """
def edit_door_append_probe(door: EditDoorBuffer, env: typing.SupportsInt | typing.SupportsIndex, edits: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], tiles: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> bool:
    """
    TEST-ONLY (#29 P7e.3): append host rows to an env of the door as a producer would (keys must continue its seq). False: nothing written.
    """
def edit_door_layout() -> dict:
    """
    #29 P7a.3: the compiled edit door rows: {edit_cols, packed_cols, tile_cols, src_col, region_col}.
    """
def edit_flush_layout() -> dict:
    """
    #29 P7e.3: the flush's books width and its fault codes by name.
    """
def edit_queue_contrib_probe(edits: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], tiles: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], env: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], salt: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P4c TEST-ONLY: the device edit queue's per-tile contribution kernel (the noise and the rounding) on host arrays.
    """
def eos_energy_books_sum(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_ambient: typing.Any = None, thermal_solid: typing.Any = None, gas_energy: typing.Any = None, t_amb_raw: typing.SupportsInt | typing.SupportsIndex = 0) -> int:
    """
    P-M4b: S = sum(n_bulk * T) over the energy books' accountable set (!solid, !thermal_solid, !vacuum, !ambient ring), raw Q16.16^2. THE SAME C++ routine EOSSolver::step's energy brackets use. arc #54: pass gas_energy + t_amb_raw to read the SAME quantity off the conserved field instead, as sum(E - N*T_AMB_raw).
    """
def eos_kick_compression_ref(wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p_new: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, cap2_plane: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], c_max: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, adiabatic_index: typing.SupportsFloat | typing.SupportsIndex, absorb_strength: typing.SupportsFloat | typing.SupportsIndex, n_floor_solver: typing.SupportsFloat | typing.SupportsIndex, t_min: typing.SupportsFloat | typing.SupportsIndex, t_max_phys: typing.SupportsFloat | typing.SupportsIndex, u_max: typing.SupportsFloat | typing.SupportsIndex, k_drag: typing.SupportsFloat | typing.SupportsIndex = 0.0, k_drag2: typing.SupportsFloat | typing.SupportsIndex = 0.0, c_v: typing.SupportsFloat | typing.SupportsIndex = 1.0, t_amb_k: typing.SupportsFloat | typing.SupportsIndex = 290.0, gas_energy: typing.Any = None, thermal_solid: typing.Any = None, is_ambient: typing.Any = None, sponge_udamp: typing.Any = None) -> tuple:
    """
    P6.4 CPU reference: replay EOSSolver::step's kick tail in place on wind_x/wind_y (+ gas_energy, if given: the arc #54 KE brackets); cap2_plane is the per-cell (h,w) int64 velocity-cap-squared plane (Q32.32 raw) — MUST be >= 0 everywhere (a negative entry makes rad=0 > cap2 reachable, i.e. a divide by zero inside the clamp). arc #54 P-G1a: step 4c is DELETED, so `temperature` is no longer written here and slots 2/3/4/7/8 of the counter tuple are retired-and-zero (D10 keeps the layout). Returns (digest_velocity, digest_compression, u_clamp_hits, u_max_hits, 0, 0, 0, ke_drag_removed, e_drag_heat_sum, 0, 0) for this call.
    """
def eos_mg_build_parity(solver: EOSSolver, pstar: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], div_u: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, is_ambient: typing.Any = None, p_amb: typing.SupportsInt | typing.SupportsIndex = 0, sponge_sigma: typing.Any = None) -> tuple[int, str]:
    """
    S8a Path A gate PART 1c (TEST-ONLY): host mg_build_levels vs the production device build on identical inputs — returns (mismatched cell count, per-level report); 0 == bit-identical.
    """
def eos_mg_solve_ref(solver: EOSSolver, pstar: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], div_u: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, p_out: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> int:
    """
    P6.3 CPU reference: replay EOSSolver::step's pressure solve on given solve inputs; writes the solved P into p_out and returns the FNV digest (== EOSSolver.digest_helmholtz for the same inputs).
    """
def eos_nsub_reads() -> int:
    """
    #29 P5: how many max(n_sub) reads the resident EOS made (one per tick).
    """
def eos_prestage_parity(solver: EOSSolver, atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], thermal_solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> tuple[int, str]:
    """
    #29 P5 TEST-ONLY: the device pre-stage vs the host pre-stage at N = 1 -- returns (mismatch count, report); 0 == bit-identical.
    """
def eos_resident_calls() -> int:
    """
    S8a Path A: how many ticks ran the fully resident EOS chain.
    """
def eos_sl_advect_ref(wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, thermal_solid: typing.Any = None) -> int:
    """
    P6.2 CPU reference: replay EOSSolver::step's SL-advection substep chain in place on wind_x/wind_y/temperature; returns the chained FNV digest over (wy, wx). dx (tile size, m; the solver's own) is REQUIRED: the displacement is eos_transport_step_q(dt, n_sub, dx) tiles per m/s (#217 patch U).
    """
def eos_step_cuda_calls() -> int:
    """
    How many engine ticks have run the chained GPU eos.step path (P6.5 dispatch-fired telemetry).
    """
def fp_deposit_dT_wide_i64(deposit: typing.SupportsInt | typing.SupportsIndex, recip_n_q: typing.SupportsInt | typing.SupportsIndex, recip_cv: typing.SupportsInt | typing.SupportsIndex, fine_bits: typing.SupportsInt | typing.SupportsIndex = 0) -> int:
    """
    fixed_point.h deposit_dT_wide_i64: the STAGED wide chain mul128_shr(mul128_shr(deposit, recip_n_q, 16), recip_cv, 32 + fine_bits) — an int64 first operand, two floors; within one LSB of deposit_dT_wide_q16 on int32-range deposits. #78: fine_bits is the currency of a deposit in the sweep's fine heat currency, converted at the chain's final narrow (0: whole heat counts).
    """
def fp_deposit_dT_wide_q16(deposit_q: typing.SupportsInt | typing.SupportsIndex, recip_n_q: typing.SupportsInt | typing.SupportsIndex, recip_cv: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h deposit_dT_wide_q16 (P-E2b): deposit/(N*c_v) chained as one 128-bit product, narrowed ONCE to int64 (not q16) — the fix for the old two-step mul_q16->recip_mul chain's silent q16 overflow at low n_floor_heat. Both deposit sites (combustion.cpp, temperature_solver.cpp Pass 1) use this; exposed so it can be verified directly rather than re-derived.
    """
def fp_div_round(n: typing.SupportsInt | typing.SupportsIndex, d: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h div_round_q: n / d rounded half away from zero.
    """
def fp_div_trunc(n: typing.SupportsInt | typing.SupportsIndex, d: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h div_trunc_q: n / d rounded toward zero (C's /).
    """
def fp_effort_step(d: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h effort_step_q: d / 8 rounded away from zero.
    """
def fp_falloff_floor(v: typing.SupportsInt | typing.SupportsIndex, d2: typing.SupportsInt | typing.SupportsIndex, r2: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h falloff_floor_q: floor(v (1 - sqrt(d2 / r2))), -1 refused.
    """
def fp_falloff_round(v: typing.SupportsInt | typing.SupportsIndex, d2: typing.SupportsInt | typing.SupportsIndex, r2: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h falloff_round_q: v (1 - sqrt(d2 / r2)) half up, -1 refused.
    """
def fp_fine_heat_shr(x: typing.SupportsInt | typing.SupportsIndex, s: typing.SupportsInt | typing.SupportsIndex, fine_bits: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h fine_heat_shr (#78): x in a currency 2^fine_bits finer than one heat count, divided by 2^(s + fine_bits), rounded toward zero, SYMMETRIC (shr_round0_signed_i64 at s + fine_bits): s = heat_inv_shift gives a thermal solid's temperature step, s = 0 whole heat counts, s = -16 the Q16 heat currency the counters are in (exact).
    """
def fp_floordiv(n: typing.SupportsInt | typing.SupportsIndex, d: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h floordiv_q: n // d (Python's floor), d != 0.
    """
def fp_half_up(num: typing.SupportsInt | typing.SupportsIndex, den: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h half_up_q: (num + den // 2) // den, floor divisions, den > 0.
    """
def fp_isqrt(n: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h isqrt_q: floor(sqrt(n)) of an int64, exact (0 for n <= 0).
    """
def fp_isqrt_ceil(n: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h isqrt_ceil_q: ceil(sqrt(n)) of an int64.
    """
def fp_isqrt_u128(hi: typing.SupportsInt | typing.SupportsIndex, lo: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h isqrt_u128_q: floor(sqrt(hi * 2^64 + lo)), exact.
    """
def fp_isqrt_u64(n: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h isqrt_u64_q: floor(sqrt(n)) of any uint64, exact.
    """
def fp_make_recip(divisor: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    fixed_point.h make_recip: round(2^32 / divisor) as an int64, the load-time reciprocal `recip_mul` consumes. Divisor must be > 0.
    """
def fp_mul_q16_round(a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h mul_q16_round_q: a * b / 65536 rounded half away from zero (128-bit product).
    """
def fp_muldiv_floor(a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex, c: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h muldiv_floor_q: (a * b) // c for a, b >= 0, c > 0 (128-bit), -1 refused.
    """
def fp_quantize(v: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    fixed_point.h quantize: a real value -> Q16.16 int32, round-half-away-from-zero, computed in double.
    """
def fp_quantize_trace(v: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    fixed_point.h quantize_trace (#88): a real trace density -> the trace currency, round-half-away, INT64 (never narrowed), range-checked.
    """
def fp_recip_mul(x_q16: typing.SupportsInt | typing.SupportsIndex, recip: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h recip_mul: x_q16 * recip >> RECIP_SHIFT via a 128-bit intermediate (the make_recip reciprocal multiply, used for the deposit's .../c_v step).
    """
def fp_reciprocal_q16(denom_q: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h reciprocal_q16: per-cell Newton reciprocal, Q16.16 in -> Q16.16 out (int64 internally, ~1 ULP accurate; self-guards denom_q <= 0 -> 0 and floors {1,2} to 3).
    """
def fp_shr_round0(x: typing.SupportsInt | typing.SupportsIndex, s: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h shr_round0: the q16 symmetric round-toward-0 shift.
    """
def fp_shr_round0_i64(x: typing.SupportsInt | typing.SupportsIndex, s: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h shr_round0_i64: the int64 twin of shr_round0 (same symmetric round-toward-0 shift, 64-bit operand).
    """
def fp_shr_round0_signed_i64(x: typing.SupportsInt | typing.SupportsIndex, s: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h shr_round0_signed_i64: divide by 2^s for a SIGNED s. s >= 0 is shr_round0_i64 exactly; s < 0 MULTIPLIES by 2^-s, which is exact (a left shift loses nothing).
    """
def fp_shr_trunc(n: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h shr_trunc_q: n / 2^k rounded toward zero.
    """
def fp_sqrt_q16(x: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h sqrt_q16: floor(sqrt(x)) of an int64, fixed 32 trips (clamped at INT32_MAX).
    """
def fp_trace_add_sat(s: typing.SupportsInt | typing.SupportsIndex, add: typing.SupportsInt | typing.SupportsIndex) -> tuple[int, int, int]:
    """
    fixed_point.h trace_add_sat (#88 P2): the counted int32 saturation, from above only. Returns (stored, clip, neg_hits): a sum above TRACE_CELL_MAX stores it and returns the excess as clip; a negative sum is COUNTED (neg_hits = 1) and stored unclamped.
    """
def fp_trace_density_shr(s: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h trace_density_shr (#88): a density sum over trace planes (coefficients Q16 per real unit) -> a Q16 fraction, s >> TRACE_SHIFT.
    """
def fp_trace_from_bulk(q: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h trace_from_bulk (#88): a bulk-count mass entering a trace plane, q * 2^TRACE_FINE_BITS.
    """
def get_bulk_flux_backend() -> bool:
    """
    True if the bulk donor-cell flux backend is set to GPU (P6.1: flag only until the P6.5 engine dispatch).
    """
def get_combustion_backend() -> bool:
    """
    True if the combustion pass currently runs on the GPU.
    """
def get_eos_step_backend() -> bool:
    """
    True iff run_substeps will dispatch eos.step to the GPU chain (all four EOS kernel-surface backend flags are on).
    """
def get_fire_backend() -> bool:
    """
    True if the fire pass currently runs on the GPU.
    """
def get_kick_compression_backend() -> bool:
    """
    True if the EOS kick+compression tail is flagged for the GPU.
    """
def get_mg_solve_backend() -> bool:
    """
    True if the EOS multigrid pressure solve is flagged for the GPU.
    """
def get_radiation_backend() -> bool:
    """
    True if the radiation sweep currently runs on the GPU.
    """
def get_sl_advection_backend() -> bool:
    """
    True if the EOS SL-advection pass is flagged for the GPU.
    """
def get_swarm_backend() -> bool:
    ...
def get_temperature_backend() -> bool:
    """
    True if the temperature pass currently runs on the GPU.
    """
def get_water_backend() -> bool:
    """
    True if the water pass currently runs on the GPU.
    """
def horde_flood_probe(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex, arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], arg6: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], arg7: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    #29 P7d, TEST-ONLY: the HORDE stage's flood (one block per env) on device masks tile_ok (N, h, w) uint8 into dist (N, h, w) int32 (device addresses).
    """
def host_sync_log_begin() -> None:
    ...
def host_sync_log_end() -> list[str]:
    ...
def object_tables_layout() -> dict:
    """
    #29 P7.0b: the compiled object tables layout (object_tables_layout.h): {fingerprint}.
    """
def philox32_4x32_10(ctr0: typing.SupportsInt | typing.SupportsIndex, ctr1: typing.SupportsInt | typing.SupportsIndex, ctr2: typing.SupportsInt | typing.SupportsIndex, ctr3: typing.SupportsInt | typing.SupportsIndex, key0: typing.SupportsInt | typing.SupportsIndex, key1: typing.SupportsInt | typing.SupportsIndex) -> tuple[int, int, int, int]:
    """
    philox32 (arc #63 P1): the vendored Philox-4x32-10 block function -- out = F(ctr[4], key[2]), 10 rounds, integer-only (cpp/src/philox32.h).
    """
def philox32_4x32_10_np(ctr: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32], key: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32]) -> numpy.typing.NDArray[numpy.uint32]:
    """
    philox32 (arc #63 P1): vectorised philox32_4x32_10 over (n,4) ctr / (n,2) key uint32 arrays -> (n,4) uint32 out. Bulk-sweep path only, same block function as philox32_4x32_10.
    """
def philox32_angle_q16(w: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    philox32 (arc #63 P1): (w * TWO_PI_Q16) >> 32 -- Q16.16 angle in [0, 2*pi).
    """
def philox32_below(w: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    philox32 (arc #63 P1): (w * n) >> 32 -- Lemire multiply-shift integer in [0, n), never % or float.
    """
def philox32_uniform_q16(w: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    philox32 (arc #63 P1): w >> 16 -- Q16.16 uniform in [0, 1).
    """
def radiation_sweep_cuda_calls() -> int:
    """
    How many GPU sweeps (radiation_sweep_step) have completed in this process — the P4 gate's dispatch-fired telemetry.
    """
def resident_burn_list() -> tuple[int, int, int, int, int]:
    """
    #29 P7e.2: (counts, lists, n_env, h, w) -- the device addresses of the last resident tick's burn-through list (the fire's (N,) int counter, the (N, h*w) int arrival-order lists), for the device sort.
    """
def resident_capture_abort() -> None:
    """
    #29 P5 TEST-ONLY: end a capture a failed tick left open.
    """
def resident_capture_begin() -> None:
    """
    #29 P5 TEST-ONLY: begin capturing the tick stream (global mode).
    """
def resident_capture_end() -> dict:
    """
    #29 P5 TEST-ONLY: end the capture; returns {segments, nodes, sync_points}.
    """
def resident_ignition_probe(material: typing.Annotated[numpy.typing.ArrayLike, numpy.int8], table: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> numpy.typing.NDArray[numpy.int32]:
    """
    #29 P4a TEST-ONLY: the resident ignition-plane gather on host arrays.
    """
def resident_stage_counters() -> dict:
    """
    #29 P4a: how many times each resident tail stage ran (and did work) in this process.
    """
def set_bulk_flux_backend(use_cuda: bool) -> None:
    """
    Switch the bulk donor-cell flux to the GPU (True) or CPU (False). P6.1: flag only — the engine dispatch lands in P6.5.
    """
def set_combustion_backend(use_cuda: bool) -> None:
    """
    Switch PhysicsRunner's combustion pass (CombustionSolver.step) to the GPU (True) or CPU (False).
    """
def set_fire_backend(use_cuda: bool) -> None:
    """
    Switch PhysicsEngine's fire pass (FireSimulation.step) to the GPU (True) or CPU (False).
    """
def set_kick_compression_backend(use_cuda: bool) -> None:
    """
    Switch the EOS kick+compression tail to the GPU (True) or CPU (False). No dispatch site consumes this until P6.5 wires eos.step's GPU path.
    """
def set_mg_solve_backend(use_cuda: bool) -> None:
    """
    Switch the EOS multigrid pressure solve to the GPU (True) or CPU (False). No dispatch site consumes this until P6.5 wires eos.step's GPU path.
    """
def set_radiation_backend(use_cuda: bool) -> None:
    """
    Switch PhysicsEngine's radiation sweep (step 2b of step_tail) to the GPU twin (True) or RadiationSweep.run (False).
    """
def set_sl_advection_backend(use_cuda: bool) -> None:
    """
    Switch the EOS SL-advection pass to the GPU (True) or CPU (False). No dispatch site consumes this until P6.5 wires eos.step's GPU path.
    """
def set_swarm_backend(use_cuda: bool) -> None:
    """
    arc #63 P3: run the host-path swarm step on the per-call CUDA wrapper.
    """
def set_temperature_backend(use_cuda: bool) -> None:
    """
    Switch PhysicsEngine's temperature pass to the GPU (True) or CPU (False).
    """
def set_water_backend(use_cuda: bool) -> None:
    """
    Switch PhysicsEngine's water pass to the GPU (True) or CPU (False).
    """
def sin_q16(a: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Q2-LIFT: pure-integer sin on Q16.16 radians (output Q16.16 in [-65536, 65536]; accuracy pinned for |a| <= 4*pi, any int32 defined).
    """
def sky_exchange_step(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], o2_idx: typing.SupportsInt | typing.SupportsIndex, inert_idx: typing.SupportsInt | typing.SupportsIndex, sky_mask: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], o2_frac_q: typing.SupportsInt | typing.SupportsIndex, lambda_q: typing.SupportsInt | typing.SupportsIndex, sky_flux: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> None:
    """
    sky-exchange: per-tick composition relaxation of sky-connected air toward ambient at fixed local N_total (O2 up / inert down); sky_flux[gas] accumulates the actual applied delta (conservation rail).
    """
def swarm_branch_tally(reset: bool = False, species: str = 'larva') -> dict:
    """
    arc #63 P3 (v4.7 G-d): the CPU loop's per-branch counts of the species law (the counts BEFORE the reset when reset=True). The device never counts.
    """
def swarm_cpu_calls() -> int:
    """
    arc #63 P3: how many CPU swarm law calls with work ran in this process.
    """
def swarm_cuda_calls() -> int:
    """
    arc #63 P3: how many per-call CUDA swarm law calls with work ran.
    """
def swarm_larva_felt_T(T: typing.Any, x_q: typing.SupportsInt | typing.SupportsIndex, y_q: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    arc #63 P3: the larva's RAW reader (ruling 11): T[y >> 16, x >> 16]; ValueError outside the grid.
    """
def swarm_species_step(species: str, columns: collections.abc.Sequence, high_water: typing.Any, temperature: typing.Any, solid: typing.Any, env: typing.Any, tables: typing.Any, death_rec: typing.Any) -> int:
    """
    arc #63 P3: the species law's CPU entry (THE reference) on caller arrays: columns = the 9 ROSTER columns (n_env, max_units) in place; high_water (n_env,) int32; temperature (n_env, h, w) int32; solid (n_env, h, w) bool; env (n_env, ENV_WORDS) uint32; tables (n_env, T_sp) int64; death_rec (n_env, max_units, DEATH_WORDS) int32 (only [0, hw[e]) written). Returns the death count.
    """
def swarm_wrap_heading_q16(h: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    arc #63 P3: the C++ twin of swarm_fixed.wrap_heading_q16 (int64 in).
    """
def tick_end_clears(n_env: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], active: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    #29 P6.4b: the tick-end clears on the (N, h, w) planes (tick stream).
    """
def tick_stream_ptr() -> int:
    """
    #29 P5: the address of THE tick stream (cudaStream_t) every resident launch core runs on.
    """
def topology_probe_seam(tables: UnitTablesBuffer, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, ops: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], n_ops: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], tiles: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P6b.1 TEST-ONLY: seal / unseal / can_seal / inject / extract / transfer / _aperture_blocked / inject_gas_n_vec / floordiv_q / muldiv_floor_q per op; returns (results (N, K, 9), books).
    """
def topology_probe_tile_changed(applier: FieldApplierBuffer, tables: UnitTablesBuffer, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P6.4b TEST-ONLY: material = m; on_tile_changed, per staged record.
    """
def unit_apply_orders(store: UnitStoreBuffer, tables: UnitTablesBuffer, requests: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> tuple:
    """
    #29 P6.2: slot 1b' (unit_orders.apply_requests) on the device store.
    """
def unit_burn_sort(store: UnitStoreBuffer, applier: FieldApplierBuffer, counts: typing.SupportsInt | typing.SupportsIndex, lists: typing.SupportsInt | typing.SupportsIndex, stride: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7e.2: slot 9's burn-through list (arrival order on the device) ranked row-major into the effect list as DESTROY_WALL records. Returns (coverage, faults, refused).
    """
def unit_burst_walls(store: UnitStoreBuffer, tables: UnitTablesBuffer, applier: FieldApplierBuffer, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7e.2: slot 9b's scan (material, solid, is_vacuum, is_ambient, atmosphere) and top-K as DESTROY_WALL records. Returns (coverage, faults, refused).
    """
def unit_charges_tick(tables: UnitTablesBuffer, charges: ObjectTableBuffer, effects: ..., tick_row: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P7b: charges.tick -- the due charges as PAYLOAD records, then the prune.
    """
def unit_convert_zombies(store: UnitStoreBuffer, tables: UnitTablesBuffer) -> tuple:
    """
    #29 P6.2: slot 10's ruleset tail (convert_marines_to_zombies) on the device store.
    """
def unit_damage_probe(store: UnitStoreBuffer, tables: UnitTablesBuffer, ring: UnitEventRingBuffer, packets: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P6.3 TEST-ONLY: the device apply_packet on the store's rows.
    """
def unit_door_uses(tables: UnitTablesBuffer, requests: UnitRecordListBuffer, doors: ObjectTableBuffer) -> tuple:
    """
    #29 P6b.1: slot 1b's USE requests flip the door table's want_open, in request-list order; returns (changed, faults, refused, dropped).
    """
def unit_draws_layout() -> dict:
    """
    #29 P7.0a: the compiled draws vocabulary (unit_draws_layout.h): {fingerprint}.
    """
def unit_events_layout() -> dict:
    """
    #29 P6.3: the compiled event ring layout (unit_events_layout.h): {width, row_cap, fingerprint}.
    """
def unit_expire_orders(store: UnitStoreBuffer, tables: UnitTablesBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P6.2: slot 10' (unit_orders.expire) on the device store.
    """
def unit_exposure_decay(store: UnitStoreBuffer, tables: UnitTablesBuffer) -> tuple:
    """
    #29 P6.2: slot 1c (exposure.decay_units) on the device store.
    """
def unit_flight_plane(tables: UnitTablesBuffer, solid: typing.SupportsInt | typing.SupportsIndex, material: typing.SupportsInt | typing.SupportsIndex, out: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P7e.1: THE FLIGHT PLANE on the device (combat.flight_plane): out (N, h, w) bool = solid | mat_blocks_thrown[material]; returns (changed, faults, refused, dropped).
    """
def unit_gas_rows(store: UnitStoreBuffer, tables: UnitTablesBuffer, ring: UnitEventRingBuffer, gas: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, claims: typing.Any, bite: typing.SupportsInt | typing.SupportsIndex, count: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P6.3: slot 9c3 (apply_teargas_blind, apply_poison_dose, arc #63 P5a's apply_bite_dose, P5b's apply_corpse_infestation, poison.tick) on the device store.
    """
def unit_grenades(tables: UnitTablesBuffer, grenades: ObjectTableBuffer, effects: ..., tick_row: typing.SupportsInt | typing.SupportsIndex, solid: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P7b: slot 2's grenades -- the flight (#215: the wall walk and the bounces over the device (N, h, w) solid plane), the fuse-outs as PAYLOAD records.
    """
def unit_heat_damage(store: UnitStoreBuffer, tables: UnitTablesBuffer, ring: UnitEventRingBuffer, heat: typing.SupportsInt | typing.SupportsIndex, rad_flux: typing.SupportsInt | typing.SupportsIndex, atmosphere: typing.SupportsInt | typing.SupportsIndex, temperature: typing.SupportsInt | typing.SupportsIndex, solid: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, claims: typing.Any) -> tuple:
    """
    #29 P6.3: slot 9c (exchange.apply_environmental_damage; #215: with the body temperature's air read) on the device store.
    """
def unit_ignition(store: UnitStoreBuffer, tables: UnitTablesBuffer, ptrs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7.0a: slot 9d (combat.apply_temperature_ignition) on (N, h, w) planes (temperature, material, flammable, wall_hp, solid, is_vacuum, gas, fire, ignition_armed); fire and armed in place. Returns (coverage, faults, refused).
    """
def unit_integrate_turn(store: UnitStoreBuffer, tables: UnitTablesBuffer) -> tuple:
    """
    #29 P6.2: slot 1a (unit_impulse.integrate_all) on the device store.
    """
def unit_kit_draws_probe(keys: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], args: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P7.0a TEST-ONLY: the device draws on synthetic (key, salt, id, index).
    """
def unit_kit_falloff_probe(args: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P7.0a TEST-ONLY: falloff_floor_q / falloff_round_q on the device.
    """
def unit_kit_sight_probe(solid: typing.SupportsInt | typing.SupportsIndex, screen: typing.SupportsInt | typing.SupportsIndex, n_env: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, pairs: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P7.0a TEST-ONLY: the device has_los (both forms) on random pairs.
    """
def unit_lib_probe(store: UnitStoreBuffer, tables: UnitTablesBuffer, op: typing.SupportsInt | typing.SupportsIndex, a0: typing.SupportsInt | typing.SupportsIndex = 0, a1: typing.SupportsInt | typing.SupportsIndex = 0, a2: typing.SupportsInt | typing.SupportsIndex = 0, a3: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P6.2 TEST-ONLY: the device unit library on the store's rows.
    """
def unit_match_done(store: UnitStoreBuffer, tables: UnitTablesBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P6.4a: the done kernel (MatchRules.is_terminal / result) -> (N, 4) int64: terminal, outcome (0 none, 1 win, 2 draw), winner team, living teams.
    """
def unit_melee(store: UnitStoreBuffer, tables: UnitTablesBuffer, effects: FieldApplierBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7a.1: slot 4a (combat.process_melee_intents) on the device store, its blows appended to the effect list; returns (coverage, faults, refused, dropped records).
    """
def unit_movement(store: UnitStoreBuffer, tables: UnitTablesBuffer, material: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P6.2m: slot 3 (Simulation._update_player_movement, the continuous path) on the device store; `material` is a device pointer to the (N, h, w) int8 GameMap.material planes.
    """
def unit_movement_probe(op: typing.SupportsInt | typing.SupportsIndex, args: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], material: typing.Annotated[numpy.typing.ArrayLike, numpy.int8], mobility: typing.Annotated[numpy.typing.ArrayLike, numpy.int64]) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P6.2m TEST-ONLY: the walk's library functions on synthetic rows.
    """
def unit_requests_layout() -> dict:
    """
    #29 P6.4a: the compiled request list layout (unit_requests_layout.h): {width, row_cap, fingerprint}.
    """
def unit_rounds(store: UnitStoreBuffer, tables: UnitTablesBuffer, effects: FieldApplierBuffer, rounds: ObjectTableBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex, solid: typing.SupportsInt | typing.SupportsIndex, material: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, compact: bool = True) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7a.2: slot 2''s march (Simulation._march_rounds) over the round table; returns (coverage, faults, refused, dropped records, rounds lost).
    """
def unit_shooting(store: UnitStoreBuffer, tables: UnitTablesBuffer, effects: FieldApplierBuffer, rounds: ObjectTableBuffer, noise: ObjectTableBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex, solid: typing.SupportsInt | typing.SupportsIndex, material: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7a.2: slot 4's shooting (combat.process_shooting, the continuous path) on the device store; returns (coverage, faults, refused, dropped records, rounds lost, noises lost).
    """
def unit_spray_cone_probe(solid: typing.SupportsInt | typing.SupportsIndex, n_env: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, args: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], box_max: typing.SupportsInt | typing.SupportsIndex, max_out: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7a.3 TEST-ONLY: combat.spray_cone_tiles' device form on synthetic cones.
    """
def unit_sprays(store: UnitStoreBuffer, tables: UnitTablesBuffer, effects: ..., cells: UnitRecordListBuffer, solid: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, box_max: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7a.3: slot 4b (combat.process_sprays) on the device store; returns (coverage, faults, refused WEGO bursts, dropped records + cells).
    """
def unit_stamp(store: UnitStoreBuffer, tables: UnitTablesBuffer, scratch: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], cap: typing.SupportsInt | typing.SupportsIndex, planes: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, rows_only: bool) -> tuple[tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]], tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]]:
    """
    #29 P7e.2: slot 6 from the unit store -- the stamp rows (rows_only) or the rows, the stamp kernel and the atmosphere refill. Returns ((row counts), (refill counts)).
    """
def unit_status_tick(store: UnitStoreBuffer, tables: UnitTablesBuffer, ring: UnitEventRingBuffer) -> tuple:
    """
    #29 P6.3: slot 2b (status.tick_statuses) on the device store.
    """
def unit_store_layout() -> dict:
    """
    #29 P6.1 / P6.2: the compiled unit store and tables layouts (unit_store_layout.h, unit_tables_layout.h): {width, fingerprint, valid, exposure, exposure_width}.
    """
def unit_table_appends(tables: UnitTablesBuffer, requests: UnitRecordListBuffer, grenades: ObjectTableBuffer, charges: ObjectTableBuffer) -> tuple:
    """
    #29 P7b: slot 1b's THROW requests appended to the grenade table, its accepted plants to the charge table, in request-list order.
    """
def unit_verbs(store: UnitStoreBuffer, tables: UnitTablesBuffer, requests: UnitRecordListBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex, spans: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], live: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], flight: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P6.4a: slot 1b (Simulation._consume_direct_intents) on the device store; returns (coverage, faults, refused plants, dropped requests).
    """
def unit_verbs_probe(store: UnitStoreBuffer, tables: UnitTablesBuffer, kind: typing.SupportsInt | typing.SupportsIndex, tick: typing.SupportsInt | typing.SupportsIndex, base: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
    """
    #29 P6.4a TEST-ONLY: the hand verbs' gates on the store's rows.
    """
def unit_verbs_tables(store: UnitStoreBuffer, tables: UnitTablesBuffer, requests: UnitRecordListBuffer, tick_row: typing.SupportsInt | typing.SupportsIndex, doors: ..., door_tiles: ..., charges: ..., flight: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64], numpy.typing.NDArray[numpy.int64]]:
    """
    #29 P7e.4: slot 1b with the door spans and the live charges built on the device from the tables (the device conductor's form).
    """
def unit_wind_impulse(store: UnitStoreBuffer, tables: UnitTablesBuffer, gas: typing.SupportsInt | typing.SupportsIndex, wind_x: typing.SupportsInt | typing.SupportsIndex, wind_y: typing.SupportsInt | typing.SupportsIndex, solid: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex) -> tuple:
    """
    #29 P6.2: slot 9c2 (exchange.apply_wind_force) on the device store; the planes are device pointers to (N, ...) arrays in the resident dtypes.
    """
def water_substeps_resident(d_depth: typing.SupportsInt | typing.SupportsIndex, d_vx: typing.SupportsInt | typing.SupportsIndex, d_vy: typing.SupportsInt | typing.SupportsIndex, d_floor: typing.SupportsInt | typing.SupportsIndex, d_atm: typing.SupportsInt | typing.SupportsIndex, d_solid: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, wdt: typing.SupportsFloat | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, damping: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, k_p: typing.SupportsFloat | typing.SupportsIndex, v_max: typing.SupportsFloat | typing.SupportsIndex, depth_eps: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    S8a Path B: water substep loop resident on device buffers (no per-substep transfer). Device pointers are CuPy .data.ptr uintptr_t.
    """
ATMOSPHERE_FIXEDPOINT: bool = True
ATMOSPHERE_FP_ONE: int = 65536
ATMOSPHERE_FP_SHIFT: int = 16
CAP_SHIFT_MAX: int = 12
CAP_SHIFT_MIN: int = -16
EDIT_FIELD_ORDINALS: dict = {'atmosphere': 0, 'fire': 1, 'gas': 2, 'gas_energy': 3, 'heat': 4, 'smoke': 5, 'water_depth': 6, 'wave_source': 7}
EDIT_QUEUE_LAYOUT: dict = {'dep': 0, 'clip': 16, 'gas_max': 16, 'ge_field_edit': 32, 'ge_field_edit_wave': 34, 'ge_field_edit_atm': 36, 'noise': 38, 'fail': 39, 'tiles': 40, 'slots': 41, 'edit_cols': 16, 'tile_cols': 4, 'env_cols': 4}
E_CEILING_TOP_Q: int = 1048575999
E_FINE_BITS: int = 11
E_INV_TOP_GAME: int = 15996
E_TABLE_SIZE: int = 4000
E_TABLE_TOP_MAX: int = 17592186044416
HAS_CUDA: bool = True
L_CHANNELS: int = 3
L_FINE_BITS: int = 37
L_TABLE_TOP_MAX: int = 17592186044416
PHILOX32_TWO_PI_Q16: int = 411775
RADIATION_SWEEP_CNT_SLOTS: int = 20
RS_BAD_HEAT_ABSORB: int = 4
RS_SLOT_BAD_AMBIENT: int = 1
RS_SLOT_BAD_EXTINCTION: int = 0
RS_SLOT_BAD_LIGHT_EXTINCTION: int = 5
RS_SLOT_BAD_SCALARS: int = 2
RS_SLOT_LIGHT_ABSORB: int = 9
RS_SLOT_LIGHT_EMIT: int = 6
RS_SLOT_LIGHT_MAX_STREAM: int = 19
RS_SLOT_LIGHT_MIN_STREAM: int = 18
RS_SLOT_LIGHT_RING_IN: int = 12
RS_SLOT_LIGHT_RING_OUT: int = 15
RS_SLOT_MAX_STREAM: int = 4
RS_SLOT_MIN_STREAM: int = 3
SWARM_CONSTANTS: dict = {'AI_DEAD': 0, 'AI_RUN': 1, 'AI_EAT': 2, 'AI_COMA': 3, 'FIRST_UNIT_ID': 1, 'PI_Q16': 205887, 'HEADING_PERIOD_Q16': 411774, 'N_COLUMNS': 9, 'ENV_TICK': 0, 'ENV_SEED_LO': 1, 'ENV_SEED_HI': 2, 'ENV_WORDS': 3, 'LAUNCH_TICK': 0, 'LAUNCH_SEED_LO': 1, 'LAUNCH_SEED_HI': 2, 'LAUNCH_HW': 3, 'LAUNCH_FIXED_WORDS': 4, 'DEATH_UNIT_ID': 0, 'DEATH_POS_X': 1, 'DEATH_POS_Y': 2, 'DEATH_HEADING': 3, 'DEATH_WORDS': 4, 'ROW_SPECIES': 0, 'ROW_ENV': 1, 'ROW_SLOT': 2, 'ROW_UNIT_ID': 3, 'ROW_POS_X': 4, 'ROW_POS_Y': 5, 'ROW_HEADING': 6, 'ROW_WORDS': 7, 'SWEEP_DIST_Q16': 65536, 'SWARM_CNT_SLOTS': 2, 'ROSTER_SIZES': {'pos_x': 4, 'pos_y': 4, 'heading': 4, 'hp': 4, 'ai_state': 4, 'dist_walked': 4, 'unit_id': 4, 'faction': 4, 'valid': 1}}
SWARM_LARVA_BRANCHES: tuple = ('slot_invalid', 'coma_held', 'coma_to_run', 'run_to_coma', 'alarm_on', 'alarm_off', 'pref_plus', 'pref_minus', 'pref_none_equal', 'pref_none_blocked', 'sweep_solid', 'sweep_off_grid', 'tumble_taken', 'tumble_not_taken', 'turn_by_coin', 'turn_toward', 'turn_away', 'probe_solid', 'probe_edge', 'moved', 'died', 'survived', 'wrap_neg_remainder')
SWARM_SPECIES_LAWS: list = [{'name': 'larva', 'table_width': 14, 'field': 'temperature', 'field_reduction': 'center', 'branches': ('slot_invalid', 'coma_held', 'coma_to_run', 'run_to_coma', 'alarm_on', 'alarm_off', 'pref_plus', 'pref_minus', 'pref_none_equal', 'pref_none_blocked', 'sweep_solid', 'sweep_off_grid', 'tumble_taken', 'tumble_not_taken', 'turn_by_coin', 'turn_toward', 'turn_away', 'probe_solid', 'probe_edge', 'moved', 'died', 'survived', 'wrap_neg_remainder')}]
SWARM_TABLES_FINGERPRINT: str = 'f633976cd7dcea4d75fa818419fcc6897910efe3dc3033acbb6cec88f82d1ebf'
TRACE_CELL_MAX: int = 2147483647
TRACE_FINE_BITS: int = 8
TRACE_SHIFT: int = 24
WATER_FIXEDPOINT: bool = True
WATER_FP_ONE: int = 65536
WATER_FP_SHIFT: int = 16
WAVE_FIXEDPOINT: bool = True
WAVE_FP_ONE: int = 65536
WAVE_FP_SHIFT: int = 16
WIND_FIXEDPOINT: bool = True
WIND_FP_ONE: int = 65536
WIND_FP_SHIFT: int = 16
