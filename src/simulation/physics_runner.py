"""PhysicsRunner — per-tick wrapper around the C++ ``breach_physics`` solvers.

Lifted from ``main.py`` (the inline ``PhysicsRunner`` shim from before
this migration) and reconciled with the parameter-binding side channel
in ``game.py:Physics._init_solvers``.

Fire is the signed-logistic intensity FEEDBACK model
(fire_design_proposal §2/§3/§5): the cellular spread is deleted (spread is
now radiation -> heat -> temperature -> ignition), and the per-tile
life/death + own-tile plume pressure deposit are bound from
``config.toml`` ``[physics.fire]`` (the ``FIRE_*`` module constants are the
fallbacks when a key is absent).

``step(gmap, sim_time)`` advances all physics by ``sim_time`` seconds
(usually one game tick = 1 / CFG.clock.ticks_per_second). The IMEX
substep loop is unchanged from the legacy. Returns the list of
``(y, x)`` coordinates where fire burned through walls — caller is
responsible for calling ``gmap.destroy_wall(y, x)`` on each.
"""
from __future__ import annotations

import math

import numpy as np

from config import CFG
from simulation import water_fixed   # S1: water_depth Q16.16 quantize helpers


# ---------------------------------------------------------------------------
# S8a Path B — GPU residency opt-in (docs/cuda_s8a_residency_spec_2026-07-19).
# Process-global, default OFF: with it off, ``PhysicsRunner.step`` takes the
# EXACT prior CPU/per-call path and CuPy is never imported. ``--resident``
# (tools/run_on_cuda.py) + the S8a gate flip it on; the runner then lazily puts
# each GameMap into residency mode (:meth:`GameMap.enable_residency`) on its
# first resident tick and dispatches to :meth:`PhysicsRunner._step_resident`.
# ---------------------------------------------------------------------------
_RESIDENCY_ENABLED = False


def set_residency(on: bool) -> None:
    """Turn GPU field residency on/off (process-global, default OFF)."""
    global _RESIDENCY_ENABLED
    _RESIDENCY_ENABLED = bool(on)


def residency_enabled() -> bool:
    return _RESIDENCY_ENABLED


# ---------------------------------------------------------------------------
# Fire feedback parameter defaults (fire_design_proposal §2/§3/§5). Cellular
# spread is GONE — spread is now radiation -> heat -> temperature -> ignition
# (apply_temperature_ignition). These drive the signed-logistic life/death of an
# already-lit tile. Bound from config [physics.fire]; these constants are the
# fallback when a key is absent. Erik tunes them live.
# ---------------------------------------------------------------------------
FIRE_K_GROW         = 4.0    # logistic growth gain (1/s)
FIRE_K_DIE          = 2.0    # decay rate when starved/cold (1/s)
FIRE_T_EXT          = 350.0  # extinction temperature (~ignition_temp + 50)
FIRE_T_SPAN         = 150.0  # width of the `hot` ramp above T_ext
FIRE_FUEL_REF       = 60.0   # wall_hp normaliser: F = clamp01(wall_hp/fuel_ref)
# Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md): o2f is LINEAR
# in the local O2 mole fraction X = Σn_o2/Σn_total, with an extinction limit.
FIRE_O2_FRAC_EXT    = 0.13   # X_ext: flame-extinction O2 mole fraction (0 = pure proportional)
FIRE_O2_FRAC_AMB    = 0.21   # X_amb: ambient O2 mole fraction (per-map: reads [ambient] o2_frac)
FIRE_P_MIN          = 0.60   # RETIRED (see o2_frac_ext/amb) — was the smoothstep low edge
FIRE_P_FULL         = 1.00   # RETIRED — was the smoothstep full edge
FIRE_I_MIN          = 0.02   # snap-to-zero extinguish floor
FIRE_K_WIND_FAN     = 0.5    # (1 + k_wind_fan*W) fans growth (firestorm); TUNE vs wind scale
FIRE_K_WIND_STRIP   = 0.5    # W*(1-I)*I blows out small fires (crossover); TUNE vs wind scale
FIRE_PRESSURE_GAIN  = 0.15   # own-tile plume overpressure gain (1/s)
FIRE_P_EXPAND_REF   = 1.30   # self-limiting plume saturation ceiling
FIRE_SMOKE_EMISSION = 0.8    # smoke produced per step by fire
FIRE_WALL_DAMAGE    = 0.4    # HP damage to wall per step while burning (the burn-out brake)


# ---------------------------------------------------------------------------
# Water pipe-model parameter defaults (engine/07 §2, water plan W2). Bound from
# config [physics.water] by _bind_water_params; these constants are the
# fallback when a key is absent. NOTE: `dx` has no constant here — the solver's
# CFL bound and gradients need the LEVEL's physical tile size, so it lazy-binds
# from ``gmap.tile_size_m`` on the first _step_water call (never assumed).
# ---------------------------------------------------------------------------
WATER_G         = 9.81   # m/s^2
WATER_DAMPING   = 1.0    # 1/s pipe friction (prototype-validated: fluid_test.py)
WATER_K_P       = 0.0    # pressure head m/atm — 0 == head OFF (W4 turns it on)
WATER_V_MAX     = 8.0    # m/s velocity clamp (paired with the C++ outflow limiter)
WATER_DEPTH_EPS = 1e-5   # m snap-to-zero (kills denormal creep)
WATER_CEILING_H = 2.5    # m air column == the solver's h_ref CFL reference
WATER_RATIO_CAP = 1.5    # max per-tick isothermal compression ratio (W3)
WATER_FLOOD_EPS = 0.05   # m air column below which a cell counts FLOODED (W3)
WATER_BOIL_RATE = 0.02   # m/s flash-boil sink in near-vacuum (W5)
WATER_BOIL_P_THRESH = 0.3  # atmosphere below this boils (W5; pressure-keyed)
WATER_STEAM_YIELD = 4.0  # steam density per metre boiled (W5)
WATER_GAMMA_R   = 2.0    # 1/s ripple damping (W6a — visual-only surface wave)
WATER_H_CAP     = 0.25   # m deep-water cap: c^2 = g*min(depth, h_cap) (W6a)
WATER_K_AMP     = 0.5    # ripple amplitude clamp |ripple| <= k_amp*depth (W6a)
WATER_K_SPLASH  = 2.0    # wave_p -> ripple_v splash gain (W6a; pure feel dial)


class PhysicsRunner:
    """Wraps the C++ atmosphere / smoke / fire solvers for one game session."""

    def __init__(self, breach_physics):
        """Build the solver instances and bind tunable params from config.

        Parameters
        ----------
        breach_physics
            The compiled ``breach_physics`` pybind11 module (passed in
            rather than imported here so test harnesses can swap it).
        """
        bp = breach_physics
        self.bp = bp

        # CUDA-S2 LIVE: the fire->heat ray cast (cast_fire_heat) can run on the GPU
        # when the raycaster backend flag is on (set via bp.set_raycaster_backend,
        # flipped by tools/run_on_cuda). The flag only EXISTS on the CUDA build —
        # the CPU build's bindings define no backend setters/getters — so cache a
        # query that is a constant False on the CPU build. cast_fire_heat reads
        # this per tick to pick the GPU `cuda_raycaster_cast` per-source path (heat
        # bit-identical to the CPU cast) vs the default CPU cast_source_directional.
        _get_ray_backend = getattr(bp, "get_raycaster_backend", None)
        self._raycaster_on_cuda = (
            _get_ray_backend if callable(_get_ray_backend) else (lambda: False))

        # EOS P6.9b: the combustion pass can run on the GPU when the combustion
        # backend flag is on (bp.set_combustion_backend). Same idiom as the
        # raycaster flag — the setter/getter only EXIST on the CUDA build, so
        # cache a query that is a constant False on the CPU build. Flag-off
        # (default) is the EXACT prior CPU CombustionSolver.step call.
        _get_comb_backend = getattr(bp, "get_combustion_backend", None)
        self._combustion_on_cuda = (
            _get_comb_backend if callable(_get_comb_backend) else (lambda: False))

        # PhysicsEngine (Patch 1 S3) owns the solver instances. The runner uses
        # its solvers (engine.<solver>) instead of constructing them itself —
        # same objects, same calls, bit-identical. engine.<solver> returns a
        # reference to the held C++ instance (reference_internal), so the param
        # binds and step() calls below act on the engine's solvers. The per-tick
        # orchestration moves INTO engine.step() in S4.
        self.engine = bp.PhysicsEngine()

        # AtmosphereSolver: wave + IMEX diffusion. Same params as legacy
        # _init_solvers; main.py's shim was missing max_source_per_step.
        self.atmos = self.engine.atmos
        self.atmos.c                   = float(CFG.physics.wave_c)
        self.atmos.damping             = float(CFG.physics.wave_damping)
        self.atmos.transfer            = float(CFG.physics.wave_transfer)
        self.atmos.d_atm               = float(CFG.physics.d_atm)
        self.atmos.feed_rate           = float(CFG.physics.source_feed_rate)
        self.atmos.breach_rate         = float(CFG.physics.breach_rate)
        self.atmos.max_source_per_step = float(
            getattr(CFG.physics, 'max_source_per_step', 0.5))
        # Global scale on per-cell wave-absorption (4a — lossy wave boundary).
        self.atmos.absorb_strength = float(
            getattr(CFG.physics, 'wave_absorb_strength', 8.0))

        # SmokeDynamics.
        self.smoke = self.engine.smoke
        self.smoke.d_smoke              = float(CFG.physics.d_smoke)
        self.smoke.advection_rate       = float(CFG.physics.advection_rate)
        self.smoke.wind_diffusion_scale = float(CFG.physics.wind_diffusion_scale)
        # (vent_hops / sink_strength binds DELETED — EOS refactor P3,
        # decisions.md #3: the BFS breach sink-pull is gone; venting is
        # native to the compressible solver.)

        # FireSimulation — signed-logistic intensity FEEDBACK (fire_design_proposal
        # §2/§3/§5). Cellular spread is gone: spread is radiation -> heat ->
        # temperature -> ignition (apply_temperature_ignition). This step is the
        # life/death of an already-lit tile: grow when hot + fuelled + pressurised,
        # decay/blow-out otherwise; deposit a self-limiting own-tile plume into
        # `atmosphere` (smoke pushed OUTWARD); burn the wall through (fuel brake).
        # All tunables bound from config [physics.fire] (FIRE_* are the fallbacks).
        fire_cfg = getattr(CFG.physics, "fire", None)

        def _fp(key, default):
            return float(getattr(fire_cfg, key, default))

        self.fire = self.engine.fire
        self.fire.params.k_grow         = _fp("k_grow", FIRE_K_GROW)
        self.fire.params.k_die          = _fp("k_die", FIRE_K_DIE)
        self.fire.params.fire_T_ext     = _fp("fire_T_ext", FIRE_T_EXT)
        self.fire.params.fire_T_span    = _fp("fire_T_span", FIRE_T_SPAN)
        self.fire.params.fuel_ref       = _fp("fuel_ref", FIRE_FUEL_REF)
        # Continuous-O2 law dials. o2_frac_amb is a per-MAP value (the level's
        # authored [ambient] o2_frac); bound to the 0.21 fallback here and
        # refreshed per-map in _ambient_args when an ambient config is present.
        self.fire.params.o2_frac_ext    = _fp("o2_frac_ext", FIRE_O2_FRAC_EXT)
        self.fire.params.o2_frac_amb    = _fp("o2_frac_amb", FIRE_O2_FRAC_AMB)
        # P_min/P_full RETIRED from the sustain law (continuous-O2 law); left
        # wired so old configs/bindings that still set them do not hard-error.
        self.fire.params.P_min          = _fp("P_min", FIRE_P_MIN)
        self.fire.params.P_full         = _fp("P_full", FIRE_P_FULL)
        self.fire.params.I_min          = _fp("I_min", FIRE_I_MIN)
        self.fire.params.k_wind_fan     = _fp("k_wind_fan", FIRE_K_WIND_FAN)
        self.fire.params.k_wind_strip   = _fp("k_wind_strip", FIRE_K_WIND_STRIP)
        self.fire.params.fire_pressure_gain = _fp(
            "fire_pressure_gain", FIRE_PRESSURE_GAIN)
        # p_expand_ref: RETIRED as the plume's self-limiting gate
        # (eos-p3fix-thermal-ceiling — see FireParams::T_FLAME_MAX,
        # fire_simulation.h). Left wired so old configs don't hard-error;
        # the C++ side no longer reads it.
        self.fire.params.p_expand_ref   = _fp("p_expand_ref", FIRE_P_EXPAND_REF)
        self.fire.params.smoke_emission = _fp("smoke_emission", FIRE_SMOKE_EMISSION)
        self.fire.params.wall_damage    = _fp("wall_damage", FIRE_WALL_DAMAGE)
        # Q16.16 scale of the temperature field (== HEAT_SCALE). Keep C++/config
        # in lockstep so T = temperature/temp_scale matches ignition_temp units.
        thermal_cfg = getattr(CFG.physics, "thermal", None)
        self.fire.params.temp_scale = float(
            getattr(thermal_cfg, "TEMP_SCALE", 65536))

        # TemperatureSolver (engine/06 §1–§2): turns the per-tick `heat` deposit
        # into the persistent `temperature` field on solids (§1 conversion), then
        # spreads it by CONDUCTION (§2). The conduction pass keys faces off the
        # NO_FACE sentinel; bind it from config so Python (the per-tile
        # face_shift bake) and C++ never disagree. Ambient cooling (§3) + unit
        # damage (§4) land in later passes.
        self.temperature = self.engine.temperature
        thermal = getattr(CFG.physics, "thermal", None)
        self.temperature.no_face = int(getattr(thermal, "NO_FACE", 63))
        # Ambient cooling dials (§3.3): interior vs vacuum-exposed decay shifts
        # and the atmosphere threshold below which a 4-neighbour counts as
        # space-facing. Bound from config so the burn-out tuning lives in one
        # place. Cooling relaxes ΔT toward 0 (T_ambient == 0): T -= T >> shift.
        self.temperature.cool_shift = int(getattr(thermal, "COOL_SHIFT", 5))
        self.temperature.cool_shift_vacuum = int(
            getattr(thermal, "COOL_SHIFT_VACUUM", 3))
        self.temperature.o2_vacuum_thresh = float(
            getattr(thermal, "o2_vacuum_thresh", 0.3))
        # EOS refactor P2 (docs/eos_refactor_design.md §4, §9): gas-T dials —
        # the wind->displacement rate for the semi-Lagrangian gas advection
        # pre-pass, the gas heat-capacity constant for the ΔT=ΔE/(N·c_v)
        # radiation deposit, and its independent N-divisor floor.
        self.temperature.gas_advection_rate = float(
            getattr(thermal, "gas_advection_rate", 900.0))
        self.temperature.c_v = float(getattr(thermal, "c_v", 1.0))
        # n_floor_heat CHECKED against the v2.4 single-tick criterion and
        # KEPT at 0.05 (eos-p3fix-thermal-ceiling — derivation in
        # temperature_solver.h/config.toml; the stacked-firestorm case is
        # bounded by the counted T_MAX_PHYS rail, not the floor).
        self.temperature.n_floor_heat = float(
            getattr(thermal, "n_floor_heat", 0.05))
        # T_MAX_PHYS (v2.4, PROVISIONAL — Erik review at P5): ONE constant,
        # wired to every solver that deposits/writes T (rationale:
        # cpp/src/eos_solver.h; config: [physics.thermal]).
        self._t_max_phys = float(getattr(thermal, "T_MAX_PHYS", 16000.0))
        self.temperature.T_MAX_PHYS = self._t_max_phys

        # --- K2: sim-side fire heat ray source (proposal §1) ------------------
        # Fire is a DETERMINISTIC heat source cast IN THE SIM (not the renderer).
        # Each burning tile becomes a short-range heat `LightSource`; we cast the
        # whole fire source list with the C++ raycaster into `gmap.heat` at the
        # START of step(), BEFORE the TemperatureSolver, so this tick's fire heat
        # converts to temperature this same tick. Heat-only: we pass scratch light
        # buffers (the render glow is a separate later step) and `heat_atten` so
        # the deposit is occluded per K1 (a wall blocks the fire's heat beyond it).
        #
        # Own Raycaster instance (headless — the renderer owns a separate one for
        # the light pass; this one only ever fills `heat`). coarse_cluster reuses
        # the shipped clustering when many tiles burn (a firestorm casts from a
        # coarse grid, not every tile). Determinism: fixed ray count, fixed angles
        # (no jitter / RNG), fixed row-major source order, integer saturating-add.
        self.raycaster = self.engine.raycaster
        fire_cfg = getattr(CFG.physics, "fire", None)
        self.raycaster.coarse_cluster = int(
            getattr(fire_cfg, "coarse_cluster", 3))
        self.k_fire_heat = float(getattr(fire_cfg, "k_fire_heat", 9.0))
        self.fire_ray_count = int(getattr(fire_cfg, "fire_ray_count", 8))
        self.fire_range_base = float(getattr(fire_cfg, "range_base", 2.0))
        self.fire_range_per_i = float(getattr(fire_cfg, "range_per_intensity", 3.0))
        self.fire_intensity_base = float(getattr(fire_cfg, "intensity_base", 0.3))
        self.fire_intensity_per_i = float(
            getattr(fire_cfg, "intensity_per_intensity", 0.7))
        col = getattr(fire_cfg, "color", [1.0, 0.45, 0.12])
        self.fire_color = (float(col[0]), float(col[1]), float(col[2]))
        # Throwaway light buffers for the heat-only cast. The march REQUIRES
        # light_rgb / light_dx / light_dy (it writes the RGB/direction channels
        # unconditionally), but fire's visual glow is a later step, so we discard
        # them. Allocated lazily on first cast (we don't know the grid size here)
        # and zeroed each pass so the discarded float accumulators can't grow
        # unbounded over a long session. `smoke_glow` is passed as None (skip).
        self._fire_scratch_rgb = None
        self._fire_scratch_dx = None
        self._fire_scratch_dy = None
        # S2b: dequantized-gas float scratch for the fire-light heat cast (the
        # raycaster's gas optics are float; gmap.gas is int32 Q16.16). Lazy alloc.
        self._fire_gas_f = None

        # EOSSolver (EOS refactor P3, docs/eos_refactor_design.md §3): the
        # compressible Kwatra pressure-evolution solver. REPLACES the
        # AtmosphereSolver wave+diffuse dispatch in run_substeps below (atmos
        # is retained on the engine only for any still-bound isolated GPU test
        # entry points — see physics_engine.cpp's GPU guards). Bound from
        # [physics.eos]; c_max/S/N_SUB_MAX/CFL_ADV/N_FLOOR_SOLVER/T_AMB_K/C/
        # gamma are the design's PINNED constants (docs/eos_refactor_
        # decisions.md 2026-07-10) — defaults on the C++ struct already match;
        # config only overrides where a key is present so a bare install still
        # gets the pinned values.
        self.eos = self.engine.eos
        eos_cfg = getattr(CFG.physics, "eos", None)

        def _ep(key, default):
            return float(getattr(eos_cfg, key, default))

        self.eos.c_max          = _ep("c_max", self.eos.c_max)
        # dx is NOT a config constant — it lazy-binds from the level's
        # tile_size_m on the first step() call below (the WaterSolver.dx
        # precedent; the design's c_max=300 m/s and its overflow budget are
        # both derived at the LEVEL's physical tile size, not a config guess).
        self.eos.S              = int(
            getattr(eos_cfg, "S", self.eos.S))
        self.eos.N_SUB_MAX      = int(
            getattr(eos_cfg, "N_SUB_MAX", self.eos.N_SUB_MAX))
        self.eos.CFL_ADV        = _ep("CFL_ADV", self.eos.CFL_ADV)
        self.eos.N_FLOOR_SOLVER = _ep("N_FLOOR_SOLVER", self.eos.N_FLOOR_SOLVER)
        self.eos.T_AMB_K        = _ep("t_amb_k", self.eos.T_AMB_K)
        self.eos.C              = _ep("C", self.eos.C)
        # ingress-lint: "adiabatic_index" (not "gamma") avoids colliding with
        # the banned RNG distribution-method name test_ingress_lint.py scans
        # for (numpy's Generator.gamma() — an unrelated collision; this is a
        # plain config attribute get/set, never a random draw).
        self.eos.adiabatic_index = _ep("adiabatic_index", self.eos.adiabatic_index)
        self.eos.absorb_strength = float(
            getattr(eos_cfg, "absorb_strength", self.eos.absorb_strength))
        self.eos.T_MIN           = _ep("T_MIN", self.eos.T_MIN)
        # v2.4 rails (PROVISIONAL — Erik review at P5): T_MAX_PHYS shares
        # [physics.thermal]'s one constant (wired above); U_MAX is the
        # solver's own [physics.eos] dial.
        self.eos.T_MAX_PHYS      = self._t_max_phys
        self.eos.U_MAX           = _ep("U_MAX", self.eos.U_MAX)

        # CombustionSolver (EOS refactor P4, docs/eos_refactor_design.md §5):
        # burns fuel against the REAL local O2, once per tick, right after
        # the EOS solver materializes P/N/T. Bound from [physics.combustion];
        # defaults on the C++ struct already match, config only overrides
        # where a key is present (the eos-block precedent above).
        self.combustion = self.engine.combustion
        comb_cfg = getattr(CFG.physics, "combustion", None)

        def _cp(key, default):
            return float(getattr(comb_cfg, key, default))

        self.combustion.burn_rate = _cp("burn_rate", self.combustion.burn_rate)
        # Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md §2.3):
        # demand = burn_rate*I*o2f*dt. o2_frac_ext/amb are the SAME law the fire
        # logistic uses (bound from [physics.fire] so there is one source of
        # truth); o2_thresh_burn is now only an epsilon skip-floor. o2_frac_amb
        # is refreshed per-map in _ambient_args (the level's [ambient] o2_frac).
        fire_cfg_c = getattr(CFG.physics, "fire", None)
        self.combustion.o2_frac_ext = float(
            getattr(fire_cfg_c, "o2_frac_ext", FIRE_O2_FRAC_EXT))
        self.combustion.o2_frac_amb = float(
            getattr(fire_cfg_c, "o2_frac_amb", FIRE_O2_FRAC_AMB))
        self.combustion.o2_thresh_burn = _cp(
            "o2_thresh_burn", self.combustion.o2_thresh_burn)
        self.combustion.H_fuel = _cp("H_fuel", self.combustion.H_fuel)
        self.combustion.soot_yield = _cp("soot_yield", self.combustion.soot_yield)
        # v2.5 (P5.1 stoichiometric fuel consumption, design §5 v2.5 /
        # decisions #17): wall_hp consumed per unit N_O2 burned — THE
        # ember-lifetime dial. Quantized once per step in C++ like the
        # other per-step scalars.
        self.combustion.fuel_per_o2 = _cp(
            "fuel_per_o2", self.combustion.fuel_per_o2)
        self.combustion.o2_thresh_breathe = _cp(
            "o2_thresh_breathe", self.combustion.o2_thresh_breathe)
        # v2.4 rail: the SAME [physics.thermal].T_MAX_PHYS constant as the
        # thermal + EOS solvers (one ceiling in the system).
        self.combustion.T_MAX_PHYS = self._t_max_phys

        # Gas-id lazy resolve (the `_steam_idx` precedent, _step_water below):
        # resolved BY NAME from the map's gas table on the first step() call
        # (gases.py is the single source of truth — never hardcode an index).
        self._o2_idx = None
        self._inert_n2_idx = None
        self._black_smoke_idx = None
        # BC: cached (n_gases,) int32 N_amb vector for the ambient ring clamp —
        # built once per map on the first ambient step() (the config is static;
        # rebuilt if the map/gas count changes). None until built / on space maps.
        self._ambient_n_amb = None
        # sky-exchange conservation rail (design §1.3): the (n_gases,) int64
        # sky_flux accumulator — Σ actual applied ΔN per plane, this tick. Cleared
        # at the top of every tick that runs the pass (so a test/telemetry reader
        # sees a single tick's exchange); lazily sized to the gas count. None until
        # the first sky-active tick / on space + dormant maps.
        self._sky_flux = None

        # WaterSolver (engine/07 §2, water plan W2): the pipe model that
        # advances gmap.water_depth. Params are bound through a METHOD (not
        # inline here) so a future config-reload hook can re-call it; all
        # [physics.water] keys are restart-bound today (engine/12 §5). `dx`
        # lazy-binds from the level's tile_size_m on the first _step_water
        # call. DORMANT-SAFE: with zero water on the map _step_water early-
        # outs and the tick is bit-identical to before water existed.
        self.water = self.engine.water
        self._bind_water_params()
        # Previous-tick water-depth snapshot (lazy alloc, the _fire_scratch_*
        # pattern). Semantics (plan W2 numerics-review fix): the depth at the
        # END of the previous tick's water accounting — NOT a copy taken this
        # tick — so FieldEdit dumps (flushed before physics) and source holds
        # are each counted EXACTLY ONCE by the W3 displacement accounting.
        self._water_depth_before = None
        # W5 steam gas index: which gmap.gas slice the flash-boil puff lands
        # in. Resolved BY NAME from the gas table ONCE, on the first
        # _step_water call (never hardcode the slice index) — the same lazy
        # bind as `dx` above; gmap is not in hand at construction.
        self._steam_idx = None

    def _bind_water_params(self):
        """Bind [physics.water] onto the WaterSolver (water plan W2).

        A separate method so a future config-reload hook can re-call it (the
        keys are restart-bound today — Ctrl+R does not re-bind solver params,
        engine/12 §5). The ``WATER_*`` module constants are the fallbacks when
        a key is absent, mirroring the fire block above. ``h_ref`` — the CFL
        reference column — is the config's ``ceiling_h`` key: ONE constant for
        the air column, shared with W3's displacement accounting. ``k_p``
        ships 0.5 (pressure head ON since W4 — the live tuning dial; the
        absent-key fallback stays 0.0, head-off-safe). ``dx`` is NOT bound
        here: it
        needs the level's tile size, which only exists once a GameMap is in
        hand, so it lazy-binds on the first :meth:`_step_water` call.

        The W3 displacement accounting and the W5 flash-boil sink run
        PYTHON-side in :meth:`_step_water` (they are not solver knobs), so
        their params — ``ceiling_h`` / ``ratio_cap`` / ``flood_eps`` and
        ``boil_rate`` / ``boil_p_thresh`` / ``steam_yield`` — bind onto the
        runner itself. ``water_ceiling_h`` reads the SAME config key as
        ``h_ref`` above: one constant for the air column, two consumers.
        ``steam_yield`` is SHARED with the fire side's evaporative
        heat-boil (their lane), so heat-boil and vacuum-boil steam
        consistently.

        The W6a ripple keys — ``gamma_r`` / ``h_cap`` / ``k_amp`` /
        ``k_splash`` — are SOLVER knobs (they drive ``step_ripple``, the
        visual-only surface wave) and bind onto the solver like the pipe
        params above. ``k_splash`` is a pure feel dial awaiting Erik's
        eyeball once W6b renders the ripple.
        """
        water_cfg = getattr(CFG.physics, "water", None)

        def _fp(key, default):
            return float(getattr(water_cfg, key, default))

        self.water.g         = _fp("g", WATER_G)
        self.water.damping   = _fp("damping", WATER_DAMPING)
        self.water.k_p       = _fp("k_p", WATER_K_P)
        self.water.v_max     = _fp("v_max", WATER_V_MAX)
        self.water.depth_eps = _fp("depth_eps", WATER_DEPTH_EPS)
        self.water.h_ref     = _fp("ceiling_h", WATER_CEILING_H)
        self.water.gamma_r   = _fp("gamma_r", WATER_GAMMA_R)
        self.water.h_cap     = _fp("h_cap", WATER_H_CAP)
        self.water.k_amp     = _fp("k_amp", WATER_K_AMP)
        self.water.k_splash  = _fp("k_splash", WATER_K_SPLASH)
        self.water_ceiling_h = _fp("ceiling_h", WATER_CEILING_H)
        self.water_ratio_cap = _fp("ratio_cap", WATER_RATIO_CAP)
        self.water_flood_eps = _fp("flood_eps", WATER_FLOOD_EPS)
        self.water_boil_rate = _fp("boil_rate", WATER_BOIL_RATE)
        self.water_boil_p_thresh = _fp("boil_p_thresh", WATER_BOIL_P_THRESH)
        self.water_steam_yield = _fp("steam_yield", WATER_STEAM_YIELD)

    # ------------------------------------------------------------------
    # Per-tick step
    # ------------------------------------------------------------------
    def step(self, gmap, sim_time):
        """Advance all physics by ``sim_time`` seconds.

        IMEX scheme: explicit wave on wave_p, implicit diffusion on
        atmosphere, smoke interleaved with atmosphere (rides the
        shockwave), single fire step at full sim_time.

        Returns
        -------
        list of (int, int)
            Tile coordinates ``(y, x)`` where fire burned through a
            wall this tick. Caller should run ``gmap.destroy_wall(y, x)``
            for each (PhysicsRunner does not touch the material grid).
        """
        # S8a Path B: the GPU-resident tick (opt-in, default OFF). Same tick,
        # same arithmetic — the water substep loop + the smoke trace loop run
        # resident on persistent device buffers (killing the substep-/plane-
        # MULTIPLIED transfer tax); EOS + combustion + the tail are bracketed
        # (one D2H/H2D each). With residency off this branch is never taken and
        # CuPy is never imported.
        if _RESIDENCY_ENABLED and getattr(self.bp, "HAS_CUDA", False):
            return self._step_resident(gmap, sim_time)

        # K2: cast the fire heat pass FIRST — at the very START of the physics
        # step, BEFORE the atmosphere/smoke loop and BEFORE the TemperatureSolver
        # below. Each burning tile deposits HEAT into `gmap.heat` (Q16.16,
        # saturating-add, occluded by `heat_atten`); the TemperatureSolver then
        # converts THIS tick's fire heat to temperature, conducts and cools it,
        # and the downstream consumers (ignition, unit damage in Simulation.step)
        # read the resulting temperature/heat. Per-tick order becomes:
        #   fire heat pass -> heat buffer -> temperature convert -> conduction
        #   -> cooling -> {ignition, unit-damage} -> ... -> clear heat.
        # ADDITIVE / de-risked: the render-side ray pass (cold sources -> ~0 heat)
        # is untouched, so there is no double-count; the cellular fire spread
        # (self.fire.step below) keeps running unchanged.
        self.cast_fire_heat(gmap)

        # Water layer (engine/07 §2, water plan W2/W3): pour / flow / settle
        # the standing-water field ONCE per tick, before the atmosphere loop —
        # so the W3 displacement (water compressing the air column) lands its
        # pressure change on `atmosphere` BEFORE diffusion re-equalises it
        # this same tick, and the flooded dyn_permeability seal is in place
        # for every atmosphere/smoke substep below. Dormant-safe: with zero
        # water the method early-outs and the tick is bit-identical to before
        # the water system existed.
        self._step_water(gmap, sim_time)

        # The compressible Kwatra solver — in C++ (PhysicsEngine::run_substeps,
        # physics_engine.cpp; EOS refactor P3, design §3). REPLACES the old
        # four-decoupled-loop IMEX substep block: `self.eos` runs its own
        # internal advection-substep loop (self-advect u, advect T, donor-cell
        # O2/N2 flux every substep, substepped compression work), then the
        # Helmholtz solve ONCE per tick, then the velocity correction. The
        # TRACE gas planes advect ONCE per tick afterward (on the solver's
        # final wind), inside run_substeps itself. `gmap.wave_p` is now the
        # repurposed P_prev buffer (see eos_solver.h); the smoke breach-sink
        # BFS field is GONE (native venting replaces it — decisions.md #3).
        # dx lazy-binds from the level's tile size every tick (cheap; mirrors
        # WaterSolver.dx's bind in _step_water — the design's c_max/overflow
        # budget are both derived at this physical dx, not a config guess).
        self.eos.dx = float(gmap.tile_size_m)
        if self._o2_idx is None:
            self._o2_idx = int(gmap.gases.name_to_id["o2"])
            self._inert_n2_idx = int(gmap.gases.name_to_id["inert_n2"])
            self._black_smoke_idx = int(gmap.gases.name_to_id["smoke"])

        # BC (boundary_conditions_spec_2026-07-19): the planetside AMBIENT ring.
        # On a space map every ambient arg is None -> the C++ path is
        # byte-identical (dormancy BY BRANCH, spec §5). On an ambient map, thread
        # the ring mask, the effective pin P_amb (the shift), the per-plane N_amb
        # (the ring clamp), and the σ-sponge grid (B3b). n_amb is a (n_gases,)
        # int32 vector — zero except the two conservative bulk planes, which the
        # bulk-transport reset clamps to the level's N-primary split. Cached: the
        # config is static per map (self._ambient_n_amb keyed off gas count).
        amb = self._ambient_args(gmap)
        self.engine.run_substeps(
            gmap.wave_p, gmap.atmosphere,
            gmap.wind_x, gmap.wind_y,
            gmap.temperature,
            gmap.obstacles, gmap.solid, gmap.is_vacuum,
            gmap.dyn_permeability, gmap.dyn_wave_absorb,
            gmap.gas, gmap.gases.diffusion, gmap.gases.conservative,
            gmap.gases.decay, self._inert_n2_idx,
            sim_time,
            is_ambient=amb[0], n_amb=amb[1], p_amb=amb[2], sponge_sigma=amb[3],
            sponge_udamp=amb[4],
        )

        # EOS refactor P4 (design §5, §3.2 "step 6: combustion pass ... reads
        # settled P/N/T, feeds next tick"): burns fuel against the REAL local
        # O2, right after the EOS solver materializes P/N/T (run_substeps,
        # above) and BEFORE this tick's consumers (step_tail's fire O2 gate
        # + the ignition O2 gate, below) read O2 — so a room that just burned
        # reads its OWN depletion this same tick (no artificial 1-tick lag).
        # Its N/T mutations never re-enter this tick's already-completed
        # Helmholtz solve; they feed NEXT tick's p* = C*N_total*T instead.
        self._run_combustion(gmap, sim_time)

        # Sky exchange (docs/sky_exchange_design_2026-07-24.md): immediately AFTER
        # combustion — combustion vitiates the local O2, the sky replenishes
        # composition at fixed N_total, and the fire's NEXT-tick read sees the
        # net. Dormant (byte-identical) on space maps / sky_tau_s == 0.
        self._run_sky_exchange(gmap, sim_time)

        # Per-tick orchestration TAIL — moved into C++ in Patch 1 S4a
        # (PhysicsEngine::step_tail, physics_engine.cpp, compiled /fp:precise).
        # The three trailing PURE-SOLVER-CALL steps that used to live here —
        # the W6a ripple, the fire feedback step, and the temperature
        # heat->conduction->cooling pass — now run as one C++ call, in the same
        # order, on the engine's own solver instances. Bit-identical (no new
        # arithmetic; gated by the per-cell A/B harness). The substep loop above
        # and the water/fire-heat steps before it move in LATER S4 sub-steps.
        #
        # The tail, for reference (the order step_tail pins):
        #   1. W6a ripple — VISUAL-ONLY surface wave (canon §6, plan W6a). Runs
        #      AFTER the IMEX loop so its splash source reads the FRESH
        #      post-substep wave_p, and BEFORE the fire step. It feeds NOTHING
        #      back into transport (writes only ripple / ripple_v). The dormancy
        #      guard — skip unless water_depth.any() or ripple.any() — is
        #      reproduced inside step_tail.
        #   2. Fire feedback (fire_design_proposal §2/§3/§5) — reads the
        #      conduction-pass `temperature` (Q16.16), the SHARED wind field, and
        #      `is_vacuum`; deposits an own-tile plume into `atmosphere`; returns
        #      the burn-through wall list.
        #   3. Temperature (engine/06 §1.2 + §2 + §3) — heat->temperature
        #      conversion on solids, one conduction relaxation, then ambient
        #      cooling. Reads THIS tick's `heat` (cast at the top of step()) and
        #      updates `temperature` in place for next tick.
        # EOS P3: gas + gas_conservative added — step_tail sums the bulk
        # O2/N2 planes for the temperature Pass-1 heat-deposit divisor (the
        # real N_total, closing the P2 density-proxy TODO).
        destroyed = self.engine.step_tail(
            gmap.ripple, gmap.ripple_v, gmap.water_depth, gmap.wave_p,
            gmap.solid,
            gmap.fire, gmap.atmosphere, gmap.smoke, gmap.wall_hp,
            gmap.temperature, gmap.wind_x, gmap.wind_y,
            gmap.is_vacuum, gmap.flammable,
            gmap.heat, gmap.heat_inv_shift, gmap.face_shift,
            # THERMAL-MASS AXIS (docs/thermal_mass_axis_design_2026-07-25.md):
            # the temperature pass's per-medium mask is `thermal_solid`
            # (thermal_mass > 0), NOT the flow mask `solid` above — so a crate
            # (permeable but thermally solid) holds an object temperature
            # instead of gas the plume advects away.
            gmap.thermal_solid,
            gmap.gas, gmap.gases.conservative, self._o2_idx,
            sim_time,
            # BC: the ambient ring is wiped to ΔT=0 in the temperature pre-pass
            # (the vacuum-breach idiom); None on space maps = byte-identical.
            is_ambient=amb[0],
        )

        # NOTE: the per-tick `heat` clear does NOT live here. `heat` has a
        # SECOND consumer downstream — unit heat damage (engine/06 §4,
        # apply_environmental_damage) — which must also read the buffer before
        # it is wiped. Per proposal §6 the clear is therefore the very last
        # step of the tick, AFTER every heat reader (the C++ conversion above,
        # the Python unit-damage, and the render-glow sample). It now lives at
        # the end of Simulation.step(), following apply_environmental_damage and
        # the recorder snapshot. (STEP A originally placed the clear here, when
        # conversion was the only consumer; STEP D moves it out.)

        return destroyed

    # ------------------------------------------------------------------
    # Combustion dispatch (EOS P4) — shared by the normal + resident ticks
    # ------------------------------------------------------------------
    def _run_combustion(self, gmap, sim_time):
        """Burn fuel against the REAL local O2 (EOS P4, design §5). GPU or CPU
        dispatch (bit-identical); factored so :meth:`_step_resident` reuses the
        exact same call the normal :meth:`step` makes."""
        ignition_temp_q16 = gmap.materials.ignition_temp_q16[gmap.material].astype(
            np.int32)
        if self._combustion_on_cuda():
            # EOS P6.9b GPU dispatch (strictly additive; bit-identical to the CPU
            # CombustionSolver.step — tests/cuda_combustion_check.py). `fire` is
            # READ again (continuous-O2 law, docs/continuous_o2_law_design_2026-
            # 07-24.md §2.3): the per-claimant intensity factor I_k in the O2
            # demand. o2_frac_ext/amb are the SAME law the fire logistic uses.
            self.bp.cuda_combustion_step(
                gmap.gas, self._o2_idx, self._inert_n2_idx, self._black_smoke_idx,
                gmap.temperature, gmap.wall_hp, gmap.fire,
                gmap.flammable, gmap.solid, gmap.is_vacuum,
                ignition_temp_q16,
                sim_time, self.temperature.c_v, self.temperature.n_floor_heat,
                self.combustion.burn_rate, self.combustion.o2_thresh_burn,
                self.combustion.H_fuel, self.combustion.soot_yield,
                self.combustion.fuel_per_o2,
                self.combustion.o2_frac_ext, self.combustion.o2_frac_amb,
                self.combustion.T_MAX_PHYS,
            )
        else:
            self.combustion.step(
                gmap.gas, self._o2_idx, self._inert_n2_idx, self._black_smoke_idx,
                gmap.temperature, gmap.wall_hp, gmap.fire,
                gmap.flammable, gmap.solid, gmap.is_vacuum,
                ignition_temp_q16,
                sim_time, self.temperature.c_v, self.temperature.n_floor_heat,
            )

    # ------------------------------------------------------------------
    # Sky exchange (planetside volumetric O2) — shared by normal + resident tick
    # ------------------------------------------------------------------
    def _run_sky_exchange(self, gmap, sim_time):
        """Relax every sky-connected air tile's composition toward ambient at
        FIXED local N_total (docs/sky_exchange_design_2026-07-24.md).

        Runs on the HOST mirror ONCE per tick, immediately AFTER combustion, in
        BOTH the normal :meth:`step` and the GPU-resident :meth:`_step_resident`
        (combustion is itself a host bracket on the mirror in the resident tick,
        so this rides that bracket — one host pass, so CPU==CUDA-resident is
        bit-identical by construction). NO-OP (byte-identical to before) on space
        maps, ring-free maps, and any level with ``sky_tau_s == 0`` (dormant):
        every such case returns before touching ``gmap.gas``.
        """
        amb = getattr(gmap, "_ambient", None)
        if amb is None or not gmap.is_ambient.any():
            return                                  # space / ring-free → dormant
        tau = float(getattr(amb, "sky_tau_s", 0.0))
        if tau <= 0.0:
            return                                  # unblessed level → dormant
        # λ = quantize(dt_tick / sky_tau_s), hoisted once per tick host-side (like
        # recip_P_span). sim_time IS the tick dt. Same Q16 quantize (round-to-
        # nearest) the ring N-split uses. λ == 0 (τ ≫ one tick and tiny dt) → no-op.
        from simulation import gas_fixed
        lambda_q = int(gas_fixed.quantize_scalar(float(sim_time) / tau))
        if lambda_q == 0:
            return
        # FIXED tick-order rebuild point (Erik's determinism caveat): the mask is
        # rebuilt here, after combustion, before the pass — never opportunistically.
        mask = gmap.ensure_sky_mask()
        if not mask.any():
            return                                  # sealed box (ring but no interior sky)
        n_gases = gmap.gas.shape[0]
        if self._sky_flux is None or self._sky_flux.shape[0] != n_gases:
            self._sky_flux = np.zeros(n_gases, dtype=np.int64)
        self._sky_flux.fill(0)                       # per-tick rail (design §1.3)
        self.bp.sky_exchange_step(
            gmap.gas, self._o2_idx, self._inert_n2_idx, mask,
            int(amb.o2_frac_q), lambda_q, self._sky_flux)

    # ------------------------------------------------------------------
    # S8a Path B — the GPU-resident tick
    # ------------------------------------------------------------------
    def _water_pre_resident(self, gmap):
        """The Python-side water pre-step (lazy init + dormancy + source holds),
        lifted verbatim from :meth:`_step_water` MINUS the ``engine.step_water``
        call (the resident path runs the substep loop on device instead). Returns
        ``True`` when water is dormant this tick (skip the whole water stage)."""
        if (self._water_depth_before is None
                or self._water_depth_before.shape != gmap.water_depth.shape):
            self._water_depth_before = gmap.water_depth.copy()
            self.water.dx = float(gmap.tile_size_m)
            self._steam_idx = int(gmap.gases.name_to_id["steam"])
        before = self._water_depth_before
        if (not gmap.water_sources and not gmap.water_depth.any()
                and not before.any()):
            return True
        for (y, x, lvl) in gmap.water_sources:
            lvl_q = water_fixed.quantize_scalar(float(lvl))
            gmap.water_depth[y, x] = max(int(gmap.water_depth[y, x]), lvl_q)
        return False

    def _step_resident(self, gmap, sim_time):
        """One GPU-resident tick (S8a: Path B framework + Path A EOS residency).
        Bit-identical to the CPU/per-call tick: the water SUBSTEP loop, the
        whole EOS STAGE (advection substeps, on-device MG build + solve,
        kick/compression — docs/cuda_s8a_path_a_impl_2026-07-21.md), and the
        smoke TRACE loop all run resident on persistent device buffers — the
        Path-B EOS bracket is GONE (spec §3.3's zero mid-tick transfers, for
        real). Combustion + the tail stay BRACKETED on the mirror (S8c).
        The numpy fields are the authoritative mirror throughout — every EOS
        input is current on the mirror at the step-4 upload (the design §2
        invariant), and the step-6 batched D2H lands the EOS/trace outputs
        back on it for the brackets + combat/recorder/render (Q4 baseline).
        """
        if not gmap.residency_on():
            gmap.enable_residency()

        # -- 1. host pre-physics (on the mirror): fire heat cast, water pre-step,
        #       lazy binds, ambient args (identical to the normal step) ----------
        self.cast_fire_heat(gmap)
        self.eos.dx = float(gmap.tile_size_m)
        if self._o2_idx is None:
            self._o2_idx = int(gmap.gases.name_to_id["o2"])
            self._inert_n2_idx = int(gmap.gases.name_to_id["inert_n2"])
            self._black_smoke_idx = int(gmap.gases.name_to_id["smoke"])
        amb = self._ambient_args(gmap)
        water_dormant = self._water_pre_resident(gmap)

        # -- 2. device handles. Rung-1 uploads ONLY what the resident loops READ
        #       (targeted H2D, just-in-time) — the bracketed stages (EOS,
        #       combustion, tail) read the numpy MIRROR directly, so the full
        #       synced set need not ride to the device this rung. `from_host()`'s
        #       DEFAULT is still the full §5b always-upload set (incl.
        #       dyn_wave_absorb / dyn_light_atten / obstacles) — that is the
        #       contract Path-A inherits when it makes the brackets resident; Rung 2
        #       must NOT narrow those masks (body-shielding depends on them). --------
        dev = gmap.device_ptrs()
        h, w = gmap._h, gmap._w

        # -- 3. WATER: substep loop RESIDENT, then the W5/W3 host tail (bracket) --
        if not water_dormant:
            # H2D only the water inputs the launch core reads. floor_height is
            # static (terrain — uploaded once by enable_residency), so it is not
            # re-sent each tick; solid IS (structural edits change it).
            gmap.from_host(["water_depth", "flow_vx", "flow_vy",
                            "atmosphere", "solid"])
            n_sub = int(self.engine.water_substep_count(sim_time))
            wdt = sim_time / n_sub   # pybind casts to float32 == step_water's wdt
            self.bp.water_substeps_resident(
                dev["water_depth"], dev["flow_vx"], dev["flow_vy"],
                dev["floor_height"], dev["atmosphere"], dev["solid"],
                h, w, n_sub, wdt, gmap.tilt_x, gmap.tilt_y,
                self.water.g, self.water.damping, self.water.dx,
                self.water.k_p, self.water.v_max, self.water.depth_eps)
            gmap.to_host(["water_depth", "flow_vx", "flow_vy"])
            self.engine.step_water_tail(
                gmap.water_depth, gmap.atmosphere, gmap.solid,
                gmap.gas, self._water_depth_before, gmap.dyn_permeability,
                self._steam_idx, sim_time,
                self.water_ceiling_h, self.water_flood_eps, self.water_ratio_cap,
                self.water_boil_rate, self.water_boil_p_thresh,
                self.water_steam_yield)

        # -- 4. EOS stage FULLY RESIDENT (S8a Path A — the bracket is GONE) ------
        # Pre-upload the EOS input set from the authoritative mirror (replaces
        # Path B's post-EOS re-upload + the per-call path's internal H2D). NOTE
        # (design §2): wave_p is NOT uploaded — on device it is written (the
        # step-0 p_prev := atmosphere D2D) before any read; dyn_wave_absorb is
        # NOT uploaded — no device kernel reads it (the kick consumes the
        # host-hoisted absorb_q plane, computed FROM the mirror inside
        # run_substeps_resident — that host hoist is where body-shielding
        # lives). is_ambient IS in the list: it is NOT static — destroy_wall's
        # joins-ambient twin mutates it on a ring-adjacent breach (gamemap
        # breach_mask), exactly like is_vacuum on a space map (the PART-1b
        # gate leg caught the stale device copy). MIRROR-CURRENCY INVARIANT:
        # every field in this list is current on the mirror here (water tail,
        # FieldEdits, stamp_units, combat edits all write the mirror; nothing
        # writes it between this upload and the resident call below).
        gmap.from_host(["atmosphere", "wind_x", "wind_y", "temperature",
                        "gas", "solid", "is_vacuum", "is_ambient",
                        "dyn_permeability"])
        # The host pre-stage (all EOS reductions — they consume tick-entry
        # state) runs on the mirror inside; the device chain (substep loop,
        # div_u/N/p*, the on-device MG build, vcycle, kick, store) runs with
        # ZERO mid-tick plane transfers. Gate: tests/cuda_s8a_check.py.
        self.engine.run_substeps_resident(
            gmap.wave_p, gmap.atmosphere,
            gmap.wind_x, gmap.wind_y, gmap.temperature,
            gmap.solid, gmap.is_vacuum,
            gmap.dyn_permeability, gmap.dyn_wave_absorb,
            gmap.gas, gmap.gases.conservative,
            sim_time,
            is_ambient=amb[0], n_amb=amb[1], p_amb=amb[2],
            d_atmosphere=dev["atmosphere"], d_wave_p=dev["wave_p"],
            d_wind_x=dev["wind_x"], d_wind_y=dev["wind_y"],
            d_temperature=dev["temperature"], d_gas=dev["gas"],
            d_solid=dev["solid"], d_is_vacuum=dev["is_vacuum"],
            d_dyn_permeability=dev["dyn_permeability"],
            d_is_ambient=dev["is_ambient"] if amb[0] is not None else 0,
            d_sponge_sigma=dev["sponge_sigma"] if amb[3] is not None else 0,
            d_sponge_udamp=dev["sponge_udamp"] if amb[4] is not None else 0,
        )

        # -- 5. TRACE smoke loop + decay RESIDENT (on device) --------------------
        # Path A: NO from_host here — the device gas/wind are FRESHER than the
        # mirror (the resident EOS just wrote them, bit-identically), and the
        # masks/perm rode up in step 4's pre-upload.
        # advection_rate = 1.0f / max(eos.dx, 1e-3f) — computed in float32 to match
        # run_substeps' float expression exactly (bit-identity of the SL displacement).
        adv_rate = np.float32(1.0) / max(np.float32(self.eos.dx), np.float32(1e-3))
        self.bp.trace_smoke_resident(
            dev["gas"], dev["wind_x"], dev["wind_y"],
            dev["solid"], dev["is_vacuum"], dev["dyn_permeability"],
            dev["is_ambient"] if amb[0] is not None else 0,
            h, w, gmap.gas.shape[0], self._inert_n2_idx,
            gmap.gases.conservative, gmap.gases.diffusion, gmap.gases.decay,
            sim_time, float(adv_rate), 0.0,
        )

        # -- 6. The once-per-tick synced-set D2H (the locked Q4 decision): the
        # EOS/trace outputs land on the mirror; combustion + the tail brackets
        # read it exactly as today. RULE (design §2): a DEFAULTED to_host() is
        # forbidden in the resident tick — the device heat/fire/wall_hp (and
        # water fields on dormant ticks) are stale-by-design and would clobber
        # the authoritative mirror.
        gmap.to_host(["atmosphere", "wave_p", "wind_x", "wind_y",
                      "temperature", "gas"])

        # -- 7. COMBUSTION bracket (mirror) --------------------------------------
        self._run_combustion(gmap, sim_time)

        # -- 7b. SKY EXCHANGE bracket (mirror) — rides the combustion host bracket
        # (design finding: no device kernel; the gas planes are on the mirror here,
        # combustion just mutated them, next tick's step-4 from_host re-uploads).
        self._run_sky_exchange(gmap, sim_time)

        # -- 8. TAIL bracket: ripple (host) + fire + temperature (mirror) --------
        destroyed = self.engine.step_tail(
            gmap.ripple, gmap.ripple_v, gmap.water_depth, gmap.wave_p,
            gmap.solid,
            gmap.fire, gmap.atmosphere, gmap.smoke, gmap.wall_hp,
            gmap.temperature, gmap.wind_x, gmap.wind_y,
            gmap.is_vacuum, gmap.flammable,
            gmap.heat, gmap.heat_inv_shift, gmap.face_shift,
            gmap.thermal_solid,          # thermal-mass axis (host mirror)
            gmap.gas, gmap.gases.conservative, self._o2_idx,
            sim_time,
            is_ambient=amb[0],
        )
        # The mirror is now authoritative for every synced field (each stage wrote
        # it; the two resident loops' outputs were D2H'd to it). No final batched
        # D2H is needed — consumers read the mirror unchanged (the Q4 baseline);
        # the device set is re-uploaded whole next tick (from_host).
        return destroyed

    # ------------------------------------------------------------------
    # BC: planetside AMBIENT ring args (boundary_conditions_spec_2026-07-19)
    # ------------------------------------------------------------------
    def _ambient_args(self, gmap):
        """Return ``(is_ambient, n_amb, p_amb, sponge_sigma, sponge_udamp)``.

        On a space map (no ambient config, or no ring tiles) returns
        ``(None, None, 0, None, None)`` — every C++ ambient branch is gated on a
        non-null pointer, so the tick is byte-identical to before BC existed
        (dormancy BY BRANCH, spec §5). On an ambient map returns the live ring
        mask, the per-plane N_amb clamp vector, the effective pin P_amb (the
        shift trick's shift), the σ-sponge grid (B3b σ, ships at 0), and the
        u-damping band grid (B3c rung 2 — the real absorber, applied in the
        kick). The N_amb vector is the N-primary split (``simulation.ambient``):
        O2 -> n_o2_q, inert_N2 -> n_n2_q, 0 elsewhere.
        """
        amb = getattr(gmap, "_ambient", None)
        if amb is None or not gmap.is_ambient.any():
            return (None, None, 0, None, None)
        # Continuous-O2 law: X_amb (the mole fraction at which o2f saturates) is
        # this map's authored ambient O2 fraction — one source of truth with the
        # BC. Refresh both consumers (fire logistic + combustion) each tick;
        # o2_frac is static per map, so this is a cheap idempotent set.
        self.fire.params.o2_frac_amb = float(amb.o2_frac)
        self.combustion.o2_frac_amb = float(amb.o2_frac)
        n_gases = gmap.gas.shape[0]
        if (self._ambient_n_amb is None
                or self._ambient_n_amb.shape[0] != n_gases):
            n_amb = np.zeros(n_gases, dtype=np.int32)
            n_amb[self._o2_idx] = int(amb.n_o2_q)
            n_amb[self._inert_n2_idx] = int(amb.n_n2_q)
            self._ambient_n_amb = n_amb
        return (gmap.is_ambient, self._ambient_n_amb,
                int(amb.pin_q), gmap.sponge_sigma, gmap.sponge_udamp)

    # ------------------------------------------------------------------
    # K2: sim-side fire heat ray pass
    # ------------------------------------------------------------------
    def cast_fire_heat(self, gmap):
        """Deposit fire's radiant heat into ``gmap.heat`` (proposal §1).

        Enumerate every burning tile (``fire > 0``) in fixed ROW-MAJOR order,
        turn each into a short-range heat :class:`LightSource`, and cast the
        whole list with the C++ raycaster into ``gmap.heat`` — Q16.16,
        saturating-add, occluded per tile by ``gmap.heat_atten`` (K1). HEAT-ONLY:
        the render light buffers are throwaway scratch (fire's visual glow is a
        separate later step) and ``smoke_glow`` is skipped (None).

        Determinism (must hold — ``heat`` is sim-affecting and feeds ignition /
        unit damage downstream):

        - **Fixed ray count, fixed angles, NO RNG.** Each source uses exactly
          ``fire_ray_count`` (8) rays with ``jitter == 0``. The 8 rays are evenly
          spaced over the full circle by the C++ march; a fixed per-source phase
          (``angle_center``) derived from the tile coords rotates the fan so
          neighbouring fires don't all fire the same 8 directions — but it is a
          pure function of (row, col), never random. No ``sim.rng`` is touched.
        - **Fixed source order.** Row-major enumeration of the burning tiles.
        - **Integer saturating-add deposit.** Order-independent -> bit-identical
          across machines / runs (the property that lets ``heat`` be a CUDA
          atomicAdd later).

        Many tiles burning is handled by the raycaster's ``coarse_cluster``
        (the shipped clustering) only inside the C++ ``update_from_fire`` path;
        here we build per-tile sources in Python and rely on the small per-source
        ``max_range`` for cost (many sources x few short rays == cheap, the
        cost discipline in fire_design_notes). The cluster dial is still bound on
        the raycaster for when the source build moves into C++.

        Called at the START of :meth:`step`, BEFORE the TemperatureSolver.
        """
        # S3a: gmap.fire is int32 Q16.16. The `> 0` burning mask is exact on the
        # integer field (0 counts == unlit); the per-tile INTENSITY that feeds the
        # heat-ray range/intensity params is dequantized to real [0,1] below (a
        # LOCAL/cosmetic + heat-payload float boundary, float-OK).
        fire = gmap.fire
        # Fast out: nothing burning -> no deposit (heat stays whatever it was;
        # the sim clears it at end of tick). Mirrors the C++ early-exit.
        burning = fire > 0
        if not bool(burning.any()):
            return

        h, w = fire.shape
        # Lazily allocate / zero the throwaway light buffers (the march writes
        # RGB + direction unconditionally; we discard them). Zeroed each pass so
        # the discarded float accumulators cannot grow unbounded over a session.
        if (self._fire_scratch_rgb is None
                or self._fire_scratch_rgb.shape[:2] != (h, w)):
            self._fire_scratch_rgb = np.zeros((h, w, 3), dtype=np.float32)
            self._fire_scratch_dx = np.zeros((h, w), dtype=np.float32)
            self._fire_scratch_dy = np.zeros((h, w), dtype=np.float32)
        else:
            self._fire_scratch_rgb.fill(0.0)
            self._fire_scratch_dx.fill(0.0)
            self._fire_scratch_dy.fill(0.0)

        # S2b: gmap.gas is int32 Q16.16. The C++ raycaster's gas optics are float,
        # so DEQUANTIZE the (N,h,w) planes to a reused float32 scratch for this
        # heat-only cast. Gases never attenuate the heat channel (only material
        # heat_atten does), so the dequantized gas does not change the heat deposit
        # — the only output that survives this cast — but the buffer must be float
        # for the raycaster to read it correctly (render-irrelevant FLOAT BRIDGE).
        from simulation import gas_fixed
        if (self._fire_gas_f is None
                or self._fire_gas_f.shape != gmap.gas.shape):
            self._fire_gas_f = np.empty(gmap.gas.shape, dtype=np.float32)
        np.multiply(gmap.gas, 1.0 / gas_fixed.FP_ONE_F,
                    out=self._fire_gas_f, casting="unsafe")

        bp = self.bp
        two_pi = 2.0 * math.pi
        ray_count = self.fire_ray_count
        # CUDA-S2 LIVE: pick the per-source cast backend ONCE for this tick (a pure
        # flag read; constant False on the CPU build). When on, each source is cast
        # with bp.cuda_raycaster_cast (build_ray_list -> the GPU march); else the
        # CPU Raycaster.cast_source_directional. BOTH accumulate the source's heat
        # into the SAME gmap.heat buffer (the GPU entry uploads the running heat,
        # saturating-atomic-adds this source's deposit, downloads it back) — so the
        # per-tick CLEAR (end of Simulation.step) + per-source ACCUMULATE semantics
        # are identical, and `heat` is byte-for-byte the same as the CPU path
        # (the S2 gate proved the march; this preserves it through the live wiring).
        # The light_rgb/dir buffers also round-trip to the host each call (render-
        # only / deterministic-exempt). Argument order is identical to the CPU call,
        # with the raycaster prepended as the first positional.
        use_cuda_ray = bool(self._raycaster_on_cuda())
        # Build the source list in ROW-MAJOR order (deterministic). np.argwhere
        # yields (row, col) pairs in C order, i.e. row-major.
        ys, xs = np.nonzero(burning)
        from simulation import fire_fixed
        # S8c item 1 (fire-FPS fix): on the CUDA path, COLLECT every source and
        # issue ONE batched device cast after the loop (cuda_raycaster_cast_batch
        # concatenates build_ray_list over all sources -> one H2D of the inputs +
        # running heat plane, one march, one D2H) instead of a whole-plane round-
        # trip PER source. `heat` is BYTE-IDENTICAL to the per-source loop: they
        # differ only in atomic-deposit order, and heat's saturating integer adds
        # are order-free. The CPU path stays a per-source cast (no transfer tax to
        # amortise). Design + 3-lens critique:
        # docs/s8c_item1_fire_heat_batch_impl_2026-07-21.md.
        cuda_sources = [] if use_cuda_ray else None
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            # S3a: dequantize the Q16.16 intensity to real [0,1] for the ray params.
            intensity_fire = float(fire[yy, xx]) / fire_fixed.FP_ONE_F
            src = bp.LightSource()
            # Cast from the tile CENTRE so the 8 rays leave symmetrically.
            src.x = float(xx) + 0.5
            src.y = float(yy) + 0.5
            src.max_range = self.fire_range_base + self.fire_range_per_i * intensity_fire
            src.ray_count = ray_count          # FIXED 8 — overrides the auto count
            src.angle_spread = two_pi          # omni
            # Fixed per-source phase from the tile coords — deterministic, NOT
            # random. Rotates the 8-ray fan so adjacent fires cover complementary
            # directions; a pure function of (col, row), bit-identical everywhere.
            src.angle_center = ((xx * 7 + yy * 13) % ray_count) * (two_pi / ray_count)
            src.intensity = self.fire_intensity_base + self.fire_intensity_per_i * intensity_fire
            src.heat = self.k_fire_heat * intensity_fire   # the sim payload
            # NO dither -- heat is sim-affecting. MUST stay 0.0: build_ray_list's
            # per-source mt19937 is drawn ONLY when jitter>0, so a nonzero jitter
            # would couple sources through the RNG sequence and desync the batched
            # cast from the per-source loop (S8c design section 1).
            src.jitter = 0.0
            src.color = self.fire_color        # render-only tint (discarded here)
            if use_cuda_ray:
                # Collected for the single batched device cast after the loop.
                cuda_sources.append(src)
            else:
                self.raycaster.cast_source_directional(
                    src,
                    self._fire_scratch_rgb,
                    self._fire_scratch_dx,
                    self._fire_scratch_dy,
                    # Multi-gas march (engine/05 §6.2): pass the full gas array +
                    # per-gas tables. Gases NEVER attenuate the heat channel (only
                    # material heat_atten does), so the heat deposit — the only output
                    # that survives this cast (smoke_glow=None) — is bit-identical to
                    # the pre-multigas single-smoke call. S2b: dequantized float bridge.
                    self._fire_gas_f,
                    gmap.gases.absorption,
                    gmap.gases.scatter_albedo,
                    gmap.dyn_light_atten,
                    gmap.heat,            # <- the only output that survives the cast
                    None,                 # smoke_glow: skipped (render-only, later)
                    gmap.heat_atten,      # K1 per-tile heat occlusion
                )
        # S8c: the ONE batched device cast for ALL sources (CUDA path only). Same
        # inputs + argument order as the per-source cuda_raycaster_cast, with the
        # source LIST in place of a single source. `heat` lands back on gmap.heat
        # (the mirror), so the tick contract and every downstream heat consumer
        # (tail temperature pass + unit damage) are unchanged.
        if use_cuda_ray and cuda_sources:
            bp.cuda_raycaster_cast_batch(
                self.raycaster,
                cuda_sources,
                self._fire_scratch_rgb,
                self._fire_scratch_dx,
                self._fire_scratch_dy,
                self._fire_gas_f,
                gmap.gases.absorption,
                gmap.gases.scatter_albedo,
                gmap.dyn_light_atten,
                gmap.heat,            # <- the only synced output (heat)
                None,                 # smoke_glow: skipped (render-only, later)
                gmap.heat_atten,      # K1 per-tile heat occlusion
            )

    # ------------------------------------------------------------------
    # Water layer (engine/07 §2, water plan W2)
    # ------------------------------------------------------------------
    def _step_water(self, gmap, sim_time):
        """Advance the standing-water layer by ``sim_time`` seconds.

        Factored out of :meth:`step` (the dormancy-test enabler — a no-op
        monkeypatch here is the A/B baseline). Per tick: apply the continuous
        source holds, run the C++ pipe-model substeps, then the W5 flash-boil
        vacuum sink, then the W3 volume-displacement accounting against the
        ``before`` snapshot (isothermal P*V onto ``atmosphere`` + the flooded
        air-seal). Walls are ``gmap.solid`` — STATIC walls only; units do NOT
        block water (Erik's explicit call: water flows under feet).
        """
        if (self._water_depth_before is None
                or self._water_depth_before.shape != gmap.water_depth.shape):
            # First call (or a new map): alloc AND seed with the CURRENT depth
            # -> level-painted water is "pre-existing" (no tick-1 compression
            # spike once W3's displacement reads `before`). A FieldEdit dump
            # landing on the very first physics tick is absorbed by this seed
            # too — same semantics, intended. Also the lazy `dx` bind: the
            # solver's CFL bound and gradients need the LEVEL's physical tile
            # size, which we only meet here (never assume a default).
            self._water_depth_before = gmap.water_depth.copy()
            self.water.dx = float(gmap.tile_size_m)
            # W5 steam slice: resolved BY NAME from the map's gas table, once
            # (gases.py is the single source of truth — never hardcode the
            # steam index here).
            self._steam_idx = int(gmap.gases.name_to_id["steam"])
        before = self._water_depth_before
        # Dormant early-out: no sources, no water now, no water last tick ->
        # nothing to do (a dry ship costs ~one .any() per tick) and the whole
        # tick stays bit-identical to before the water system existed.
        if (not gmap.water_sources and not gmap.water_depth.any()
                and not before.any()):
            return
        # Continuous source holds (pipe leak / breach inflow): per-tick
        # `depth = max(depth, level_m)` — the same architectural slot as
        # wave_source feeding. Event-shaped dumps go through the FieldEdit
        # queue (flushed before physics) instead. Both are counted vs
        # `before` exactly once when W3's displacement lands. KEPT PYTHON
        # (sparse, stateful) — the array-arithmetic that follows moved into
        # C++ in Patch 1 S4c (PhysicsEngine::step_water, /fp:precise).
        # S1: water_depth is int32 Q16.16 — quantize the source level (metres)
        # to Q16.16 before the max-hold so the field stays integer.
        for (y, x, lvl) in gmap.water_sources:
            lvl_q = water_fixed.quantize_scalar(float(lvl))
            gmap.water_depth[y, x] = max(int(gmap.water_depth[y, x]), lvl_q)
        # The water-layer ARRAY ARITHMETIC — moved into C++ in Patch 1 S4c
        # (PhysicsEngine::step_water, physics_engine.cpp, compiled /fp:precise).
        # The block that used to live here — the substep-count derivation + the
        # `water.step` substep loop, the W5 flash-boil, the W3 volume
        # displacement + flooded dyn_permeability seal, and the final
        # `copyto(before, water_depth)` — now runs as one C++ call, on the
        # engine's own WaterSolver instance, BIT-IDENTICALLY.
        #
        # The precision contract (reproduced exactly in step_water):
        #   * substep count: n = max(1, int(ceil(sim_time / water.max_dt()))) in
        #     DOUBLE (the integer cliff); wdt = (float)(sim_time / n) at the
        #     water.step boundary.
        #   * W5: boiling = (atmosphere < boil_p_thresh) & (water_depth > 0) with
        #     each scalar cast to float32; boil amount = min(water_depth,
        #     (f32)(boil_rate*sim_time)) (the product in double, cast once); the
        #     steam puff (steam_yield * boiled) in float32; guarded by .any().
        #   * W3: free_before/after = max(ceiling_h - x, flood_eps) in f32; ratio
        #     = clip(free_before/free_after, 1.0/ratio_cap, ratio_cap) (clip ==
        #     min(max(.))), the low bound's reciprocal in double then cast f32;
        #     atmosphere *= ratio; flooded = free_after <= flood_eps -> seal; then
        #     before = water_depth (the copyto). The W3/W5 scalar params (config
        #     doubles) are cast to float32 INSIDE step_water at numpy's exact cast
        #     points — the bit-identity hinge. Gated by the per-cell A/B harness
        #     (0-ULP vs the post-S4b AND the pre-Patch-1 goldens).
        #
        # The water pipe params (g/damping/k_p/v_max/depth_eps/h_ref/dx) are
        # already members on the engine's WaterSolver (bound in
        # _bind_water_params), and water.dx is bound on the lazy init above — so
        # step_water re-passes none of them. `before` (the _water_depth_before
        # snapshot) is passed in and MUTATED by the final copyto (the runner keeps
        # owning it across ticks). `gmap.gas` is the (N,h,w) array; the steam puff
        # lands in slice self._steam_idx.
        # EOS refactor P3: `gmap.wave_p` arg retired (the water head reads the
        # integer `gmap.atmosphere` == P directly, no float bridge).
        self.engine.step_water(
            gmap.water_depth, gmap.flow_vx, gmap.flow_vy,
            gmap.floor_height, gmap.atmosphere,
            gmap.solid, gmap.gas, before, gmap.dyn_permeability,
            self._steam_idx, gmap.tilt_x, gmap.tilt_y, sim_time,
            self.water_ceiling_h, self.water_flood_eps, self.water_ratio_cap,
            self.water_boil_rate, self.water_boil_p_thresh,
            self.water_steam_yield,
        )

    def _step_ripple(self, gmap, sim_time):
        """Advance the VISUAL-ONLY ripple field (plan W6a, canon §6).

        A damped kick-drift surface wave riding ON TOP of ``water_depth``:
        c² = g·min(depth, h_cap) (the deep-water cap), splash-sourced from
        the fresh post-substep ``wave_p`` (gain ``k_splash`` — the pure feel
        dial), clamped to |ripple| ≤ k_amp·depth, zeroed on dry/solid. It
        NEVER feeds back into transport (the locked canon rule): the solver
        writes only ``gmap.ripple`` / ``gmap.ripple_v``; ``water_depth`` /
        ``wave_p`` / ``solid`` are read-only. Factored out of :meth:`step`
        (the ``_step_water`` precedent) so the W6a visual-only A/B test can
        no-op it.

        Dormancy: skipped when there is no water AND no leftover ripple
        anywhere. Ripple is zero wherever depth is zero by construction, so
        on a dry ship both ``.any()`` are cheap falses; the extra
        ``ripple.any()`` term lets ONE final call sweep ghost ripple to zero
        if the last wet tile drains/boils away between calls, after which
        the skip is total again.

        ONE call per tick at full ``sim_time`` — no substep loop:
        ``ripple_max_dt() = 0.5·dx/sqrt(g·h_cap)`` ≈ 106 ms at dx = 1/3,
        comfortably above any tick we use (41.7 ms at 24 tps) — the same
        derived-bound discipline as the substep counts above, with the
        substep machinery statically unnecessary.
        """
        if not gmap.water_depth.any() and not gmap.ripple.any():
            return
        # EOS refactor P3: the splash source is |P - P_prev| (design §6).
        self.water.step_ripple(gmap.ripple, gmap.ripple_v, gmap.water_depth,
                               gmap.atmosphere, gmap.wave_p, gmap.solid, sim_time)
