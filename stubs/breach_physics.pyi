"""
Breach physics engine -- C++ accelerated simulation
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['ATMOSPHERE_FIXEDPOINT', 'ATMOSPHERE_FP_ONE', 'ATMOSPHERE_FP_SHIFT', 'AtmosphereSolver', 'CAP_SHIFT_MAX', 'CAP_SHIFT_MIN', 'CombustionSolver', 'EOSSolver', 'E_INV_TOP_GAME', 'E_TABLE_SIZE', 'EmissiveTable', 'FireParams', 'FireSimulation', 'HAS_CUDA', 'LightSource', 'PhysicsEngine', 'RadiationSweep', 'Raycaster', 'SmokeDynamics', 'TemperatureSolver', 'WATER_FIXEDPOINT', 'WATER_FP_ONE', 'WATER_FP_SHIFT', 'WAVE_FIXEDPOINT', 'WAVE_FP_ONE', 'WAVE_FP_SHIFT', 'WIND_FIXEDPOINT', 'WIND_FP_ONE', 'WIND_FP_SHIFT', 'WaterSolver', 'atan2_q16', 'bulk_flux_transport', 'conduction_cell_capacity_q', 'cos_q16', 'cuda_available', 'cuda_bulk_flux_transport', 'cuda_combustion_step', 'cuda_device_info', 'cuda_eos_energy_flux', 'cuda_eos_kick_compression', 'cuda_eos_mg_solve', 'cuda_eos_sl_advect', 'cuda_fire_step', 'cuda_map_mul_q16', 'cuda_raycaster_cast', 'cuda_raycaster_cast_batch', 'cuda_smoke_step', 'cuda_spike_add1', 'cuda_temperature_step', 'cuda_water_step', 'eos_energy_books_sum', 'eos_kick_compression_ref', 'eos_mg_build_parity', 'eos_mg_solve_ref', 'eos_resident_calls', 'eos_sl_advect_ref', 'eos_step_cuda_calls', 'fp_deposit_dT_wide_i64', 'fp_deposit_dT_wide_q16', 'fp_make_recip', 'fp_quantize', 'fp_recip_mul', 'fp_reciprocal_q16', 'fp_shr_round0', 'fp_shr_round0_i64', 'fp_shr_round0_signed_i64', 'get_bulk_flux_backend', 'get_combustion_backend', 'get_eos_step_backend', 'get_fire_backend', 'get_kick_compression_backend', 'get_mg_solve_backend', 'get_raycaster_backend', 'get_sl_advection_backend', 'get_smoke_backend', 'get_temperature_backend', 'get_water_backend', 'rad_pair_budget_s', 'set_bulk_flux_backend', 'set_combustion_backend', 'set_fire_backend', 'set_kick_compression_backend', 'set_mg_solve_backend', 'set_raycaster_backend', 'set_sl_advection_backend', 'set_smoke_backend', 'set_temperature_backend', 'set_water_backend', 'sin_q16', 'sky_exchange_step', 'smoke_cliff_count', 'trace_smoke_resident', 'water_substeps_resident']
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
    def step(self, gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], o2_idx: typing.SupportsInt | typing.SupportsIndex, inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, black_smoke_idx: typing.SupportsInt | typing.SupportsIndex, temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], ignition_temp_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dt: typing.SupportsFloat | typing.SupportsIndex, c_v: typing.SupportsFloat | typing.SupportsIndex, n_floor_heat: typing.SupportsFloat | typing.SupportsIndex, thermal_solid: typing.Any = None, heat_inv_shift: typing.Any = None, heat: typing.Any = None, dem_acc: typing.Any = None, draw_r: typing.SupportsInt | typing.SupportsIndex = 1, dyn_permeability: typing.Any = None, max_claimants: typing.SupportsInt | typing.SupportsIndex = 4, gas_energy: typing.Any = None, is_ambient: typing.Any = None, t_amb_q: typing.SupportsInt | typing.SupportsIndex = 0, fire_T_ext_plane: typing.Any = None, fuel_per_o2_plane: typing.Any = None) -> None:
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
class EmissiveTable:
    def __init__(self) -> None:
        ...
    def bake(self) -> None:
        """
        Bake (or re-bake) the E° table from the current dials.
        """
    def e_bucket_of(self, T_q: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        emissive_table.h e_bucket_of: Q16.16 temperature -> bucket index.
        """
    def e_inv_q(self, phi: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        emissive_table.h e_inv_q: E°⁻¹(Φ) as a Q16.16 game temperature (the bucket's LOW edge; 0 below E°[0]; saturates at 15996 game).
        """
    def table(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        A COPY of the baked E° table (E_TABLE_SIZE int64 entries); bakes on first use and whenever a dial has moved.
        """
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
class LightSource:
    def __init__(self) -> None:
        ...
    @property
    def angle_center(self) -> float:
        ...
    @angle_center.setter
    def angle_center(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def angle_spread(self) -> float:
        ...
    @angle_spread.setter
    def angle_spread(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def color(self) -> tuple[float, float, float]:
        ...
    @color.setter
    def color(self, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
    @property
    def heat(self) -> float:
        ...
    @heat.setter
    def heat(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def intensity(self) -> float:
        ...
    @intensity.setter
    def intensity(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def jitter(self) -> float:
        ...
    @jitter.setter
    def jitter(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def max_range(self) -> float:
        ...
    @max_range.setter
    def max_range(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def ray_count(self) -> int:
        ...
    @ray_count.setter
    def ray_count(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def x(self) -> float:
        ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def y(self) -> float:
        ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class PhysicsEngine:
    def __init__(self) -> None:
        ...
    def run_substeps(self, p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_energy: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_diffusion: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas_decay: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, is_ambient: typing.Any = None, n_amb: typing.Any = None, p_amb: typing.SupportsInt | typing.SupportsIndex = 0, sponge_sigma: typing.Any = None, sponge_udamp: typing.Any = None, do_traces: bool = True, thermal_solid: typing.Any = None) -> None:
        ...
    def run_substeps_resident(self, p_prev: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], sim_time: typing.SupportsFloat | typing.SupportsIndex, is_ambient: typing.Any = None, n_amb: typing.Any = None, p_amb: typing.SupportsInt | typing.SupportsIndex = 0, d_atmosphere: typing.SupportsInt | typing.SupportsIndex = 0, d_wave_p: typing.SupportsInt | typing.SupportsIndex = 0, d_wind_x: typing.SupportsInt | typing.SupportsIndex = 0, d_wind_y: typing.SupportsInt | typing.SupportsIndex = 0, d_temperature: typing.SupportsInt | typing.SupportsIndex = 0, d_gas: typing.SupportsInt | typing.SupportsIndex = 0, d_solid: typing.SupportsInt | typing.SupportsIndex = 0, d_is_vacuum: typing.SupportsInt | typing.SupportsIndex = 0, d_dyn_permeability: typing.SupportsInt | typing.SupportsIndex = 0, d_is_ambient: typing.SupportsInt | typing.SupportsIndex = 0, d_sponge_sigma: typing.SupportsInt | typing.SupportsIndex = 0, d_sponge_udamp: typing.SupportsInt | typing.SupportsIndex = 0, thermal_solid: typing.Any = None, d_thermal_solid: typing.SupportsInt | typing.SupportsIndex = 0, d_gas_energy: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def stamp_units(self, permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_atten: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_wave_absorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dyn_light_atten: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], ys: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], xs: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], perm: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], wabsorb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], atten_r: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], atten_g: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], atten_b: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], heat_q: numpy.typing.NDArray[numpy.int32]) -> None:
        ...
    def step_tail(self, ripple: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], ripple_v: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wave_p: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], heat: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat_inv_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], face_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], thermal_solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], fuel_recip: typing.Annotated[numpy.typing.ArrayLike, numpy.int64], fire_T_ext_plane: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], o2_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], rad_net_sweep: numpy.typing.NDArray[numpy.int64], rad_flux_sweep: numpy.typing.NDArray[numpy.int64], rad_amb_sweep: numpy.typing.NDArray[numpy.int64], rad_fluence: numpy.typing.NDArray[numpy.int64], k_leak_q: typing.SupportsInt | typing.SupportsIndex, rad_amb_vacuum_q: typing.SupportsInt | typing.SupportsIndex = -1, is_ambient: typing.Any = None, rad_net: typing.Any = None, gas_energy: typing.Any = None, t_amb_q: typing.SupportsInt | typing.SupportsIndex = 0) -> list:
        ...
    def step_water(self, water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vx: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vy: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], floor_height: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], before: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], steam_idx: typing.SupportsInt | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ceiling_h: typing.SupportsFloat | typing.SupportsIndex, flood_eps: typing.SupportsFloat | typing.SupportsIndex, ratio_cap: typing.SupportsFloat | typing.SupportsIndex, boil_rate: typing.SupportsFloat | typing.SupportsIndex, boil_p_thresh: typing.SupportsFloat | typing.SupportsIndex, steam_yield: typing.SupportsFloat | typing.SupportsIndex, gas_energy: typing.Any = None, gas_conservative: typing.Any = None, thermal_solid: typing.Any = None, is_vacuum: typing.Any = None, is_ambient: typing.Any = None, t_amb_raw: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def step_water_tail(self, water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], before: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], steam_idx: typing.SupportsInt | typing.SupportsIndex, sim_time: typing.SupportsFloat | typing.SupportsIndex, ceiling_h: typing.SupportsFloat | typing.SupportsIndex, flood_eps: typing.SupportsFloat | typing.SupportsIndex, ratio_cap: typing.SupportsFloat | typing.SupportsIndex, boil_rate: typing.SupportsFloat | typing.SupportsIndex, boil_p_thresh: typing.SupportsFloat | typing.SupportsIndex, steam_yield: typing.SupportsFloat | typing.SupportsIndex, gas_energy: typing.Any = None, gas_conservative: typing.Any = None, thermal_solid: typing.Any = None, is_vacuum: typing.Any = None, is_ambient: typing.Any = None, t_amb_raw: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
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
    def emissive(self) -> EmissiveTable:
        ...
    @property
    def eos(self) -> EOSSolver:
        ...
    @property
    def fire(self) -> FireSimulation:
        ...
    @property
    def radiation(self) -> RadiationSweep:
        ...
    @property
    def raycaster(self) -> Raycaster:
        ...
    @property
    def smoke(self) -> SmokeDynamics:
        ...
    @property
    def temperature(self) -> TemperatureSolver:
        ...
    @property
    def water(self) -> WaterSolver:
        ...
class RadiationSweep:
    F_ONE: typing.ClassVar[int] = 16777216
    F_SHIFT: typing.ClassVar[int] = 24
    SHEAR: typing.ClassVar[int] = 1
    STEP: typing.ClassVar[int] = 0
    @staticmethod
    def fleck_f_solid_q24(e_table: EmissiveTable, T_q: typing.SupportsInt | typing.SupportsIndex, a_q: typing.SupportsInt | typing.SupportsIndex, his: typing.SupportsInt | typing.SupportsIndex, t_amb_q: typing.SupportsInt | typing.SupportsIndex) -> int:
        """
        The solid-branch Fleck factor (Q24) for one cell, exactly as the sweep's pre-pass forms it — the tile inspector's `f` row.
        """
    @staticmethod
    def ordinate_constants(n_ordinates: typing.SupportsInt | typing.SupportsIndex, transport: typing.SupportsInt | typing.SupportsIndex) -> list:
        """
        The checked-in per-ordinate constants as (sx, sy, x_major, s_m) tuples, for the recompute test.
        """
    def __init__(self) -> None:
        ...
    def derive_ambient(self, is_vacuum: numpy.typing.NDArray[numpy.bool], e_table: EmissiveTable, vac_level: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.int64]:
        """
        A COPY of the per-cell ambient LEVEL plane derived from vacuum/interior state (thermal model v2 R3): a vacuum cell takes vac_level, every other cell E°[0]. vac_level < 0 means E°[0] (R4, the shipped uniform answer); above E°[0] raises.
        """
    def fleck_plane(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        A COPY of the Fleck plane (Q24, (h, w)) the last run() computed.
        """
    def run(self, temperature: numpy.typing.NDArray[numpy.int32], heat_atten_q: numpy.typing.NDArray[numpy.int32], dyn_heat_atten_q: numpy.typing.NDArray[numpy.int32], heat_inv_shift: numpy.typing.NDArray[numpy.int32], thermal_solid: numpy.typing.NDArray[numpy.bool], e_table: EmissiveTable, amb_level: typing.Any, t_amb_q: typing.SupportsInt | typing.SupportsIndex, k_leak_q: typing.SupportsInt | typing.SupportsIndex, transport: typing.SupportsInt | typing.SupportsIndex, n_ordinates: typing.SupportsInt | typing.SupportsIndex, rad_net: numpy.typing.NDArray[numpy.int64], rad_flux: numpy.typing.NDArray[numpy.int64], rad_amb: numpy.typing.NDArray[numpy.int64], rad_fluence: numpy.typing.NDArray[numpy.int64], fleck_enabled: bool = True) -> None:
        """
        One tick of the sweep over all ordinates. It OVERWRITES the four int64 planes — zeroed here before the first ordinate, so they hold the last run's values until the next run and the tile inspector can read them at render time (design row 38); no caller wipe needed. transport: RadiationSweep.STEP or .SHEAR; n_ordinates: 16 or 12. fleck_enabled=False is the reference's undamped (f_plane=None) configuration, for the gates. amb_level is the PER-CELL ambient LEVEL plane (int64 (h, w), 0 <= amb <= e_table[0]) the sweep radiates against — None means E°[0] everywhere, the uniform thermal-v2-R4 configuration.
        """
    @property
    def max_stream(self) -> int:
        ...
    @property
    def min_stream(self) -> int:
        ...
class Raycaster:
    @staticmethod
    def normalize_directions(light_dx: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dy: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        ...
    def __init__(self) -> None:
        ...
    def bake_emissive_table(self) -> None:
        """
        (Re)bake the black-body E° table from the current rad_scale. Idempotent. Two owners share this one bake implementation (tests/test_emissive_table.py): this Raycaster and PhysicsEngine.emissive.
        """
    def cast_source_directional(self, source: LightSource, light_rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dx: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dy: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_absorption: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_scatter: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_atten: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], heat: typing.Any = None, smoke_glow: typing.Any = None, heat_atten: typing.Any = None) -> None:
        ...
    def emissive_table(self) -> numpy.typing.NDArray[numpy.int64]:
        """
        P-F1a: a COPY of the baked E° table (E_TABLE_SIZE INT64 entries, 4 game-units per bucket) — the oracle for the bake's tests. The table widened from int32 at P-F1a (L2-B3): its old INT32_MAX saturation above T_game ~ 1768 was a silent ceiling on the law.
        """
    @property
    def heat_cull(self) -> float:
        ...
    @heat_cull.setter
    def heat_cull(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
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
    def light_cull(self) -> float:
        ...
    @light_cull.setter
    def light_cull(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def rad_scale(self) -> float:
        ...
    @rad_scale.setter
    def rad_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def smoke_absorb_scale(self) -> float:
        ...
    @smoke_absorb_scale.setter
    def smoke_absorb_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def smoke_absorption(self) -> float:
        ...
    @smoke_absorption.setter
    def smoke_absorption(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def smoke_absorption_rgb(self) -> tuple[float, float, float]:
        ...
    @smoke_absorption_rgb.setter
    def smoke_absorption_rgb(self, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
    @property
    def smoke_scatter_albedo(self) -> tuple[float, float, float]:
        ...
    @smoke_scatter_albedo.setter
    def smoke_scatter_albedo(self, arg1: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
class SmokeDynamics:
    def __init__(self) -> None:
        ...
    def step(self, smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def advection_rate(self) -> float:
        ...
    @advection_rate.setter
    def advection_rate(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def d_smoke(self) -> float:
        ...
    @d_smoke.setter
    def d_smoke(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def wind_diffusion_scale(self) -> float:
        ...
    @wind_diffusion_scale.setter
    def wind_diffusion_scale(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class TemperatureSolver:
    def __init__(self) -> None:
        ...
    def step(self, temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat_inv_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], face_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Any = None, wind_y: typing.Any = None, dt: typing.SupportsFloat | typing.SupportsIndex = 0.0, n_bulk: typing.Any = None, thermal_solid: typing.Any = None, rad_net: typing.Any = None, rad_fluence: typing.Any = None, e_table: typing.Any = None, clamp_enabled: bool = True) -> None:
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
    def e_ring_pin_sum(self) -> int:
        ...
    @property
    def e_solid_cond_sum(self) -> int:
        ...
    @property
    def e_solid_deposit_sum(self) -> int:
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
def atan2_q16(y: typing.SupportsInt | typing.SupportsIndex, x: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Q2-LIFT: pure-integer atan2 on Q16.16 (radians out, Q16.16; range [-205887, +205887] == [-quantize(pi), +quantize(pi)]).
    """
def bulk_flux_transport(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    EOS P1: donor-cell conservative flux transport of every `gas_conservative`-flagged plane, once, on the given wind field.
    """
def conduction_cell_capacity_q(is_ts: bool, heat_inv_shift: typing.SupportsInt | typing.SupportsIndex, n_raw: typing.SupportsInt | typing.SupportsIndex, n_floor_q: typing.SupportsInt | typing.SupportsIndex, c_v_q: typing.SupportsInt | typing.SupportsIndex) -> tuple[int, int]:
    """
    temperature_solver.h conduction::cell_capacity_q -> (cap_used, cap_real). THE one capacity law both backends call.
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
def cuda_combustion_step(gas: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], o2_idx: typing.SupportsInt | typing.SupportsIndex, inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, black_smoke_idx: typing.SupportsInt | typing.SupportsIndex, temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], ignition_temp_q16: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], dt: typing.SupportsFloat | typing.SupportsIndex, c_v: typing.SupportsFloat | typing.SupportsIndex, n_floor_heat: typing.SupportsFloat | typing.SupportsIndex, burn_rate: typing.SupportsFloat | typing.SupportsIndex, o2_thresh_burn: typing.SupportsFloat | typing.SupportsIndex, H_FUEL_M: typing.SupportsFloat | typing.SupportsIndex, H_FUEL_SHIFT: typing.SupportsInt | typing.SupportsIndex, soot_yield: typing.SupportsFloat | typing.SupportsIndex, fuel_per_o2: typing.SupportsFloat | typing.SupportsIndex, o2_frac_ext: typing.SupportsFloat | typing.SupportsIndex, o2_frac_full: typing.SupportsFloat | typing.SupportsIndex, T_MAX_PHYS: typing.SupportsFloat | typing.SupportsIndex, thermal_solid: typing.Any = None, heat_inv_shift: typing.Any = None, heat: typing.Any = None, H_BED_M: typing.SupportsFloat | typing.SupportsIndex = 0.0, H_BED_SHIFT: typing.SupportsInt | typing.SupportsIndex = 0, dem_acc: typing.Any = None, draw_r: typing.SupportsInt | typing.SupportsIndex = 1, dyn_permeability: typing.Any = None, max_claimants: typing.SupportsInt | typing.SupportsIndex = 4, fire_T_ext: typing.SupportsFloat | typing.SupportsIndex = 350.0, fire_T_span: typing.SupportsFloat | typing.SupportsIndex = 180.0, hotf_cap: typing.SupportsFloat | typing.SupportsIndex = 10.0, fire_T_ext_plane: typing.Any = None, fuel_per_o2_plane: typing.Any = None) -> tuple:
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
def cuda_eos_sl_advect(wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, thermal_solid: typing.Any = None) -> int:
    """
    P6.2 isolated: run the GPU fused SL-advection substep chain in place on wind_x/wind_y/temperature (bit-identical to eos_sl_advect_ref) and return the chained FNV digest (== EOSSolver.digest_advect).
    """
def cuda_fire_step(fire: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_o2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], n_total: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wall_hp: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], flammable: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, k_grow: typing.SupportsFloat | typing.SupportsIndex, k_die: typing.SupportsFloat | typing.SupportsIndex, fire_T_ext: typing.SupportsFloat | typing.SupportsIndex, fire_T_span: typing.SupportsFloat | typing.SupportsIndex, fuel_ref: typing.SupportsFloat | typing.SupportsIndex, o2_frac_ext: typing.SupportsFloat | typing.SupportsIndex, o2_frac_full: typing.SupportsFloat | typing.SupportsIndex, I_min: typing.SupportsFloat | typing.SupportsIndex, k_wind_fan: typing.SupportsFloat | typing.SupportsIndex, k_wind_strip: typing.SupportsFloat | typing.SupportsIndex, wall_damage: typing.SupportsFloat | typing.SupportsIndex, temp_scale: typing.SupportsFloat | typing.SupportsIndex, I_cap_per_avail: typing.SupportsFloat | typing.SupportsIndex = 2.5299999713897705, o2_frac_amb: typing.SupportsFloat | typing.SupportsIndex = 0.20999999344348907, o2f_cap: typing.SupportsFloat | typing.SupportsIndex = 5.0, hotf_cap: typing.SupportsFloat | typing.SupportsIndex = 10.0, fuel_recip: typing.Any = None, fire_T_ext_plane: typing.Any = None) -> list:
    """
    P6.8 isolated: run ONE GPU fire step (re-derived — continuous-O2 mole-fraction gate) in place on fire/smoke/wall_hp (bit-identical to FireSimulation.step) and return the destroyed-walls list of (y,x) tuples. temperature is still a parameter but is READ ONLY as of P-R2 (the plume->T shim write is deleted).
    """
def cuda_map_mul_q16(in_: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], factor_q16: typing.SupportsInt | typing.SupportsIndex) -> list[int]:
    """
    S0 hello-world: out[i] = mul_q16(in[i], factor_q16) computed on the GPU via the shared toolkit; bit-identical to the CPU mul_q16.
    """
def cuda_raycaster_cast(raycaster: Raycaster, source: LightSource, light_rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dx: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dy: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_absorption: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_scatter: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_atten: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], heat: typing.Any = None, smoke_glow: typing.Any = None, heat_atten: typing.Any = None) -> None:
    """
    S2 isolated: cast one LightSource on the GPU into the pre-zeroed output fields; `heat` is bit-identical to Raycaster.cast_source_directional.
    """
def cuda_raycaster_cast_batch(raycaster: Raycaster, sources: collections.abc.Sequence[LightSource], light_rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dx: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_dy: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_absorption: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_scatter: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], light_atten: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], heat: typing.Any = None, smoke_glow: typing.Any = None, heat_atten: typing.Any = None) -> None:
    """
    S8c: cast a SEQUENCE of LightSources in ONE device march (the fire-FPS fix). `heat` is bit-identical to a per-source cuda_raycaster_cast loop (order-free saturating add). Render channels differ in float-atomic order from the per-source path and are only valid for callers that discard rgb/dir/glow (cast_fire_heat).
    """
def cuda_smoke_step(smoke: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], obstacles: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_wall: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, d_smoke: typing.SupportsFloat | typing.SupportsIndex, wind_diffusion_scale: typing.SupportsFloat | typing.SupportsIndex, advection_rate: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    S4a isolated: run the GPU smoke solver in place on one gas plane (bit-identical to SmokeDynamics.step).
    """
def cuda_spike_add1(dev_ptr: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    S8a spike: int32 in-place +1 on a raw device pointer (CuPy .data.ptr).
    """
def cuda_temperature_step(temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], heat_inv_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], face_shift: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], atmosphere: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], no_face: typing.SupportsInt | typing.SupportsIndex, o2_vacuum_thresh: typing.SupportsFloat | typing.SupportsIndex, c_v: typing.SupportsFloat | typing.SupportsIndex, n_floor_heat: typing.SupportsFloat | typing.SupportsIndex, gas_advection_rate: typing.SupportsFloat | typing.SupportsIndex, t_max_phys: typing.SupportsFloat | typing.SupportsIndex, n_bulk: typing.Any = None, wind_x: typing.Any = None, wind_y: typing.Any = None, dt: typing.SupportsFloat | typing.SupportsIndex = 0.0, thermal_solid: typing.Any = None, gas_energy: typing.Any = None, t_amb_k: typing.SupportsFloat | typing.SupportsIndex = 290.0, rad_net: typing.Any = None) -> tuple:
    """
    P6.6/P-G2 isolated: run the GPU unified temperature solver in place on `temperature` (+ `gas_energy` when supplied — bit-identical to TemperatureSolver.step); returns (t_max_phys_hits, e_cond_trunc_sum, e_cond_cap_sum, cond_limit_hits, e_cool_sum, e_vac_wipe_sum, e_ring_pin_sum, e_deposit_drop_sum, e_gas_deposit_sum, e_gas_cond_sum, e_gas_rail_sum, e_solid_deposit_sum, e_solid_cond_sum, e_thermostat_sum, solid_energy_books_sum) for this call (P-E2a + P-E2b + arc #54 + P-G5; the last is a SNAPSHOT, not a per-call delta).
    """
def cuda_water_step(water_depth: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vx: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], flow_vy: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], floor_height: typing.Any = None, atmosphere: typing.Any = None, solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dt: typing.SupportsFloat | typing.SupportsIndex, tilt_x: typing.SupportsFloat | typing.SupportsIndex, tilt_y: typing.SupportsFloat | typing.SupportsIndex, g: typing.SupportsFloat | typing.SupportsIndex, damping: typing.SupportsFloat | typing.SupportsIndex, dx: typing.SupportsFloat | typing.SupportsIndex, k_p: typing.SupportsFloat | typing.SupportsIndex, v_max: typing.SupportsFloat | typing.SupportsIndex, depth_eps: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    S3 isolated: run the GPU water solver in place on water_depth/flow_vx/flow_vy (bit-identical to WaterSolver.step).
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
def eos_resident_calls() -> int:
    """
    S8a Path A: how many ticks ran the fully resident EOS chain.
    """
def eos_sl_advect_ref(wind_x: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], wind_y: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], temperature: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], solid: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], is_vacuum: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], dyn_permeability: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, n_sub: typing.SupportsInt | typing.SupportsIndex, thermal_solid: typing.Any = None) -> int:
    """
    P6.2 CPU reference: replay EOSSolver::step's SL-advection substep chain in place on wind_x/wind_y/temperature; returns the chained FNV digest (== EOSSolver.digest_advect for the same inputs).
    """
def eos_step_cuda_calls() -> int:
    """
    How many engine ticks have run the chained GPU eos.step path (P6.5 dispatch-fired telemetry).
    """
def fp_deposit_dT_wide_i64(deposit: typing.SupportsInt | typing.SupportsIndex, recip_n_q: typing.SupportsInt | typing.SupportsIndex, recip_cv: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h deposit_dT_wide_i64: the STAGED wide chain mul128_shr(mul128_shr(deposit, recip_n_q, 16), recip_cv, 32) — an int64 first operand, two floors; within one LSB of deposit_dT_wide_q16 on int32-range deposits.
    """
def fp_deposit_dT_wide_q16(deposit_q: typing.SupportsInt | typing.SupportsIndex, recip_n_q: typing.SupportsInt | typing.SupportsIndex, recip_cv: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    fixed_point.h deposit_dT_wide_q16 (P-E2b): deposit/(N*c_v) chained as one 128-bit product, narrowed ONCE to int64 (not q16) — the fix for the old two-step mul_q16->recip_mul chain's silent q16 overflow at low n_floor_heat. Both deposit sites (combustion.cpp, temperature_solver.cpp Pass 1) use this; exposed so it can be verified directly rather than re-derived.
    """
def fp_make_recip(divisor: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    fixed_point.h make_recip: round(2^32 / divisor) as an int64, the load-time reciprocal `recip_mul` consumes. Divisor must be > 0.
    """
def fp_quantize(v: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    fixed_point.h quantize: a real value -> Q16.16 int32, round-half-away-from-zero, computed in double.
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
def get_raycaster_backend() -> bool:
    """
    Vestigial since T6 (issue #12) -- see set_raycaster_backend.
    """
def get_sl_advection_backend() -> bool:
    """
    True if the EOS SL-advection pass is flagged for the GPU.
    """
def get_smoke_backend() -> bool:
    """
    True if the smoke pass currently runs on the GPU.
    """
def get_temperature_backend() -> bool:
    """
    True if the temperature pass currently runs on the GPU.
    """
def get_water_backend() -> bool:
    """
    True if the water pass currently runs on the GPU.
    """
def rad_pair_budget_s(abs_dT_q: typing.SupportsInt | typing.SupportsIndex, his: typing.SupportsInt | typing.SupportsIndex, shift: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    raycaster.h rad_pair_budget_s: the flux limiter's per-end budget, floor(x * 2^his / 2^shift). SIGNED in `his` since M1.
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
def set_raycaster_backend(use_cuda: bool) -> None:
    """
    Vestigial since T6 (issue #12): used to switch PhysicsRunner.cast_fire_heat's fire->heat ray cast between GPU and CPU; that method is deleted, so this now sets state nothing reads. Kept because several CUDA check scripts call it unconditionally.
    """
def set_sl_advection_backend(use_cuda: bool) -> None:
    """
    Switch the EOS SL-advection pass to the GPU (True) or CPU (False). No dispatch site consumes this until P6.5 wires eos.step's GPU path.
    """
def set_smoke_backend(use_cuda: bool) -> None:
    """
    Switch PhysicsEngine's smoke pass to the GPU (True) or CPU (False).
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
def smoke_cliff_count(c4st_q: typing.SupportsInt | typing.SupportsIndex, dsmoke_q: typing.SupportsInt | typing.SupportsIndex, wds_q: typing.SupportsInt | typing.SupportsIndex, mws_q32: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Bedrock: integer smoke-CFL substep count n=ceil(4*sim_time*d_smoke_max*(1+wds*max_wind_sq)) from quantized inputs.
    """
def trace_smoke_resident(d_gas_base: typing.SupportsInt | typing.SupportsIndex, d_wx: typing.SupportsInt | typing.SupportsIndex, d_wy: typing.SupportsInt | typing.SupportsIndex, d_solid: typing.SupportsInt | typing.SupportsIndex, d_vac: typing.SupportsInt | typing.SupportsIndex, d_perm: typing.SupportsInt | typing.SupportsIndex, d_amb: typing.SupportsInt | typing.SupportsIndex, h: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsInt | typing.SupportsIndex, n_gases: typing.SupportsInt | typing.SupportsIndex, inert_n2_idx: typing.SupportsInt | typing.SupportsIndex, gas_conservative: typing.Annotated[numpy.typing.ArrayLike, numpy.bool], gas_diffusion: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], gas_decay: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], dt: typing.SupportsFloat | typing.SupportsIndex, advection_rate: typing.SupportsFloat | typing.SupportsIndex, wind_diffusion_scale: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    S8a Path B: per-tick trace-plane smoke loop + decay resident on device (no per-plane transfer). gas_conservative/diffusion/decay are host (N,) columns; field pointers are CuPy .data.ptr uintptr_t.
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
E_INV_TOP_GAME: int = 15996
E_TABLE_SIZE: int = 4000
HAS_CUDA: bool = True
WATER_FIXEDPOINT: bool = True
WATER_FP_ONE: int = 65536
WATER_FP_SHIFT: int = 16
WAVE_FIXEDPOINT: bool = True
WAVE_FP_ONE: int = 65536
WAVE_FP_SHIFT: int = 16
WIND_FIXEDPOINT: bool = True
WIND_FP_ONE: int = 65536
WIND_FP_SHIFT: int = 16
