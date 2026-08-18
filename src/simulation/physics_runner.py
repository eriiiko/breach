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

import warnings

import numpy as np

import temperature_scale        # P-K2: canonical game-T -> Kelvin map accessor
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


# (residency_enabled() DELETED - audit Patch A / A9, 2026-08-04: an accessor
# with no caller anywhere. The _RESIDENCY_ENABLED global it read is live and
# is set by set_residency() above and read directly at the dispatch site.)


# ---------------------------------------------------------------------------
# Fire feedback parameter defaults (fire_design_proposal §2/§3/§5). Cellular
# spread is GONE — spread is now radiation -> heat -> temperature -> ignition
# (apply_temperature_ignition). These drive the signed-logistic life/death of an
# already-lit tile. Bound from config [physics.fire]; these constants are the
# fallback when a key is absent. Erik tunes them live.
# ---------------------------------------------------------------------------
FIRE_K_GROW         = 4.0    # logistic growth gain (1/s) — TEMPO only since P-R3
FIRE_K_DIE          = 2.0    # decay rate when starved/cold (1/s)
# CAPACITY LAW (P-R3, 2026-07-31 — docs/radiation_raycaster_extinction_ruling_
# 2026-07-31.md A3, on Erik's ruling R-b): the growth term's carrying capacity
# per unit availability, `I_cap = c * avail * hot`, so `I_eq ~= c*a`. THE SIZE
# DIAL — it replaces the old hardwired `(1-I)` capacity, which forced `k_die`
# to do both jobs (size AND the death wall) and left the fire 1.242x of
# headroom on `F*o2f*hot`.
FIRE_I_CAP_PER_AVAIL = 2.53  # c — capacity per unit availability
# FIRE_T_EXT: SUPERSEDED as the extinction floor by the per-tile
# `GameMap.fire_T_ext_plane` (P-R3, ruling A3 ride-along). It is now DERIVED per
# material as `ignition_temp[mat] - ignition_to_ext_delta`; the shipped global
# 350 sat ABOVE both flammable materials' ignition temps (wood 300, furniture
# 280), so a tile could ignite below its own sustain floor. Still bound (below)
# because the solver keeps it as the FALLBACK when no per-tile plane is
# supplied; the live engine always supplies one.
FIRE_T_EXT          = 350.0  # fallback extinction temperature (superseded)
FIRE_T_SPAN         = 150.0  # width of the `hot` ramp above T_ext (STAYS global)
# FIRE_FUEL_REF: SUPERSEDED as the fuel normaliser by the per-tile
# `GameMap.fuel_recip` plane (fuel-fraction axis, 2026-07-30). F is "the
# fraction of THIS tile's fuel remaining", and 60.0 is WOOD's hp — one global
# standing in for a per-material quantity, so a full-health furniture crate
# (hp 30) permanently read F = 0.5. Still bound (below) because the solver keeps
# it as the FALLBACK divisor when no per-tile plane is supplied; the live engine
# always supplies one.
FIRE_FUEL_REF       = 60.0   # fallback wall_hp normaliser (superseded, see above)
# Continuous-O2 law (docs/continuous_o2_law_design_2026-07-24.md): o2f is LINEAR
# in the local O2 mole fraction X = Σn_o2/Σn_total, with an extinction limit.
FIRE_O2_FRAC_EXT    = 0.13   # X_ext: flame-extinction O2 mole fraction (0 = pure proportional)
# FULL-RESPONSE REFERENCE SPLIT (2026-07-30): the law's denominator used to be
# X_amb, so ambient always gave o2f = 1 and the clamp made AMBIENT the ceiling —
# local O2 enrichment could never register. X_full is the separate reference at
# which o2f reaches 1 (pure O2); ambient air lands at (0.21-0.13)/(1-0.13)=0.092.
FIRE_O2_FRAC_FULL   = 1.0    # X_full: full-response reference — NOT ambient, NOT per-map
FIRE_O2_FRAC_AMB    = 0.21   # X_amb: what the ambient atmosphere IS (per-map: reads [ambient] o2_frac)
FIRE_P_MIN          = 0.60   # RETIRED (see o2_frac_ext/amb) — was the smoothstep low edge
FIRE_P_FULL         = 1.00   # RETIRED — was the smoothstep full edge
FIRE_I_MIN          = 0.02   # snap-to-zero extinguish floor
FIRE_K_WIND_FAN     = 0.5    # (1 + k_wind_fan*W) fans growth (firestorm); TUNE vs wind scale
FIRE_K_WIND_STRIP   = 0.5    # W*(1-I)*I blows out small fires (crossover); TUNE vs wind scale
# fire_pressure_gain TOMBSTONE (P-R2, 2026-07-31): dead key — the plume->T shim
# it fed (FireParams::fire_pressure_gain) was deleted; see docs/radiation_
# raycaster_extinction_ruling_2026-07-31.md A2.
FIRE_P_EXPAND_REF   = 1.30   # self-limiting plume saturation ceiling
# FIRE_SMOKE_EMISSION TOMBSTONE (P-S1, 2026-08-15): removed along with the
# `smoke_emission` key it fell back for — see the loud stale-key guard below
# and fire_simulation.h's own tombstone comment on the struct field.
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

        # P-S1 (2026-08-15, docs/smoke_single_source_asbuilt_2026-08-15.md):
        # `smoke_emission` was retired along with the ex-nihilo fire-smoke
        # scatter it drove (Erik's single-source ruling, docs/
        # smoke_single_source_design_2026-07-24.md — combustion soot,
        # [physics.combustion] soot_yield, is now the ONE fire-smoke source).
        # LOUD guard, same idiom as src/temperature_scale.py's migration
        # guards: an old config.toml still carrying the key would otherwise
        # silently do nothing (FireParams no longer has the field), and
        # whoever is tuning it would never find out why it has no effect.
        if getattr(fire_cfg, "smoke_emission", None) is not None:
            raise RuntimeError(
                "[physics.fire] still carries 'smoke_emission' — this key "
                "was retired at P-S1 (the ex-nihilo fire smoke scatter was "
                "deleted; combustion soot, [physics.combustion] soot_yield, "
                "is now the single fire-smoke source). Remove it from "
                "config.toml. See docs/smoke_single_source_asbuilt_"
                "2026-08-15.md and docs/smoke_single_source_design_"
                "2026-07-24.md."
            )

        def _fp(key, default):
            return float(getattr(fire_cfg, key, default))

        self.fire = self.engine.fire
        self.fire.params.k_grow         = _fp("k_grow", FIRE_K_GROW)
        self.fire.params.k_die          = _fp("k_die", FIRE_K_DIE)
        # CAPACITY LAW (P-R3, ruling A3): `c` — the SIZE dial. See the
        # FIRE_I_CAP_PER_AVAIL block above for what it replaced and why.
        self.fire.params.I_cap_per_avail = _fp("I_cap_per_avail",
                                               FIRE_I_CAP_PER_AVAIL)
        # fire_T_ext is the FALLBACK only since P-R3 — the live gate is the
        # per-tile `GameMap.fire_T_ext_plane` (ignition_temp - Δ), which
        # step_tail passes on every tick.
        self.fire.params.fire_T_ext     = _fp("fire_T_ext", FIRE_T_EXT)
        self.fire.params.fire_T_span    = _fp("fire_T_span", FIRE_T_SPAN)
        self.fire.params.fuel_ref       = _fp("fuel_ref", FIRE_FUEL_REF)
        # Continuous-O2 law dials. o2_frac_full is the FULL-RESPONSE reference —
        # the mole fraction at which o2f reaches 1 (pure O2) — and is deliberately
        # NOT per-map: it is a physical reference, not an atmosphere. o2_frac_amb
        # IS a per-MAP value (the level's authored [ambient] o2_frac); bound to the
        # 0.21 fallback here and refreshed per-map in _ambient_args when an ambient
        # config is present. Since the split, o2_frac_amb is no longer read by
        # either O2 law (fire logistic / combustion) — it stays as the ambient
        # record other systems and levels rely on.
        self.fire.params.o2_frac_ext    = _fp("o2_frac_ext", FIRE_O2_FRAC_EXT)
        self.fire.params.o2_frac_full   = _fp("o2_frac_full", FIRE_O2_FRAC_FULL)
        self.fire.params.o2_frac_amb    = _fp("o2_frac_amb", FIRE_O2_FRAC_AMB)
        # P_min/P_full RETIRED from the sustain law (continuous-O2 law); left
        # wired so old configs/bindings that still set them do not hard-error.
        self.fire.params.P_min          = _fp("P_min", FIRE_P_MIN)
        self.fire.params.P_full         = _fp("P_full", FIRE_P_FULL)
        self.fire.params.I_min          = _fp("I_min", FIRE_I_MIN)
        self.fire.params.k_wind_fan     = _fp("k_wind_fan", FIRE_K_WIND_FAN)
        self.fire.params.k_wind_strip   = _fp("k_wind_strip", FIRE_K_WIND_STRIP)
        # p_expand_ref: RETIRED as the plume's self-limiting gate
        # (eos-p3fix-thermal-ceiling); the plume deposit itself (the
        # fire_pressure_gain-fed shim) is DELETED at P-R2 — see
        # fire_simulation.h. Left wired so old configs don't hard-error;
        # the C++ side no longer reads it.
        self.fire.params.p_expand_ref   = _fp("p_expand_ref", FIRE_P_EXPAND_REF)
        # smoke_emission bind REMOVED at P-S1 — see the loud stale-key guard
        # above (FireParams no longer has the field to bind).
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
        # COOL-SHIFT AXIS (2026-07-30): the decay shift itself is now a
        # PER-MATERIAL column projected to `GameMap.cool_shift` and handed to
        # step_tail below. These two globals keep two live jobs: `cool_shift`
        # is the solver's fallback when no per-tile grid is supplied, and the
        # PAIR defines the vacuum-exposure discount as an OFFSET
        # (cool_shift - cool_shift_vacuum) applied to every material's own
        # shift — so "space sheds 4x faster" stays one rule and each material
        # keeps exactly ONE dial. `cool_shift_floor` clamps that subtraction;
        # it is the SAME SHIFT_MIN the material loader validates the column
        # against, bound from the one config key so the two can never disagree.
        self.temperature.cool_shift = int(getattr(thermal, "COOL_SHIFT", 5))
        self.temperature.cool_shift_vacuum = int(
            getattr(thermal, "COOL_SHIFT_VACUUM", 3))
        self.temperature.cool_shift_floor = int(getattr(thermal, "SHIFT_MIN", 2))
        self.temperature.o2_vacuum_thresh = float(
            getattr(thermal, "o2_vacuum_thresh", 0.3))
        # EOS refactor P2 (docs/eos_refactor_design.md §4, §9): gas-T dials —
        # the wind->displacement rate for the semi-Lagrangian gas advection
        # pre-pass, the gas heat-capacity constant for the ΔT=ΔE/(N·c_v)
        # radiation deposit, and its independent N-divisor floor.
        self.temperature.gas_advection_rate = float(
            getattr(thermal, "gas_advection_rate", 900.0))
        self.temperature.c_v = float(getattr(thermal, "c_v", 1.0))
        # n_floor_heat (energy-books arc, design §2.2, RULING 2026-08-17): now
        # a LOW, tunable VALUE-hygiene dial — default 0.01 (was 0.05). Its
        # stability job is gone (P-E1 closed the transport books; T_MAX_PHYS
        # is the real value backstop) — see temperature_solver.h / config.toml
        # for the full rationale.
        self.temperature.n_floor_heat = float(
            getattr(thermal, "n_floor_heat", 0.01))
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
        # the light pass; this one only ever fills `heat`). Determinism: fixed
        # ray count, fixed angles (no jitter / RNG), fixed row-major source
        # order, integer saturating-add. P-R1 (2026-07-31): the per-tile source
        # build that used to run here in Python now lives in C++
        # (Raycaster.cast_from_fire_plane / cuda_raycaster_cast_from_fire_plane,
        # see cast_fire_heat below) — coarse_cluster died with the orphaned
        # update_from_fire path it only ever fed (no production caller).
        self.raycaster = self.engine.raycaster
        fire_cfg = getattr(CFG.physics, "fire", None)
        # k_fire_heat TOMBSTONE (P-R4, 2026-08-01 — ruling A1): the painter is
        # dead. There is no per-tile one-way heat payload any more; a fire's
        # radiant transport is the antisymmetric net-T⁴ exchange below, whose
        # magnitude is set by `rad_scale` (the E° bake) and by the emitters'
        # own temperatures. Nothing reads the config key; it survives only as a
        # config comment so an old config does not hard-error.
        #
        # P-R4 dials (ruling A1.3 / A1.8), both LIVE on the raycaster:
        #   rad_scale   — the E° bake's emission calibration (heat counts per
        #                 K⁴, with σ / the 0.833 m² face / dt / the game↔Kelvin
        #                 mapping folded in). Setting it re-bakes the table.
        #   T_emit_gate — the temperature at which a NON-burning thermal solid
        #                 also starts CASTING (i.e. can radiatively lose heat).
        #                 Receivers are free: a cold crate is heated correctly
        #                 on the flame's rays whatever this is.
        self.raycaster.rad_scale = float(getattr(fire_cfg, "rad_scale", 1.0e-5))
        # P-K2: the canonical game-T -> Kelvin map (design §2/§3a), read via
        # the accessor rather than CFG directly so the phi_exp/eos_slope
        # load-time assert (temperature_scale._assert_invariants) always
        # runs on this path too. K(t) is baked from these below.
        _ts = temperature_scale.load(CFG)
        self.raycaster.kelvin_ambient = float(_ts.kelvin_ambient)
        self.raycaster.k_temp_to_kelvin = float(_ts.k_temp_to_kelvin)
        self.raycaster.T_emit_gate = float(getattr(fire_cfg, "T_emit_gate", 180.0))
        # P-F1a / v7 rule 4: RADIATION_RANGE — the emission ray's reach. A
        # STABILITY-CLASS constant, not a feel dial: at or above the grid
        # diagonal, reach-termination can never precede the world edge, so
        # "genuinely escapes" == "left the world" and the corridor leak is
        # structurally impossible. Below the floor the books stop closing, so
        # this is a HARD ERROR rather than a clamp — a silently-corrected
        # stability constant is exactly the kind of thing that gets shipped.
        _rad_range = float(getattr(fire_cfg, "RADIATION_RANGE", 320.0))
        _rad_floor = float(self.raycaster.RADIATION_RANGE_MIN)
        if _rad_range < _rad_floor:
            raise ValueError(
                f"[physics.fire] RADIATION_RANGE = {_rad_range} is below the "
                f"floor {_rad_floor} (the grid diagonal of the largest shipping "
                f"level, 128x256 -> 286.22). An emission ray that expires before "
                f"the world edge charges its residual to nobody, which reopens "
                f"the corridor leak the v7 rule-4 range floor exists to close. "
                f"Raise the key; it is not a feel dial.")
        self.raycaster.radiation_range = _rad_range
        self.raycaster.bake_emissive_table()
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
        # [physics.eos]; c_max/S/N_SUB_MAX/CFL_ADV/N_FLOOR_SOLVER/gamma are
        # the design's PINNED constants (docs/eos_refactor_decisions.md
        # 2026-07-10) — defaults on the C++ struct already match; config only
        # overrides where a key is present so a bare install still gets the
        # pinned values. T_AMB_K/C/S_EOS come from the canonical accessor
        # (_ts, [physics.temperature_scale]) instead — P-K3, design §2/§3c.
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
        # PRESSURE ARC (docs/pressure_arc_root_cause_2026-08-17.md): the MG
        # V-cycle schedule becomes config-visible. It was a C++-only default
        # "FROZEN at the MG gate" at C=2, measured there on 16²/160² scenarios
        # for 300 ticks; at real map size over long runs C=2 leaves ~0.28 atm
        # of residual per tick, which IS the storm. Config-visible so the
        # schedule can be swept and gated like any other solver constant.
        self.eos.mg_cycles      = int(
            getattr(eos_cfg, "mg_cycles", self.eos.mg_cycles))
        self.eos.mg_nu1         = int(
            getattr(eos_cfg, "mg_nu1", self.eos.mg_nu1))
        self.eos.mg_nu2         = int(
            getattr(eos_cfg, "mg_nu2", self.eos.mg_nu2))
        self.eos.mg_coarsest_sweeps = int(
            getattr(eos_cfg, "mg_coarsest_sweeps", self.eos.mg_coarsest_sweeps))
        self.eos.N_SUB_MAX      = int(
            getattr(eos_cfg, "N_SUB_MAX", self.eos.N_SUB_MAX))
        self.eos.CFL_ADV        = _ep("CFL_ADV", self.eos.CFL_ADV)
        self.eos.N_FLOOR_SOLVER = _ep("N_FLOOR_SOLVER", self.eos.N_FLOOR_SOLVER)
        # P-K3: [physics.eos] no longer carries t_amb_k/C — both are read via
        # the canonical accessor (_ts, loaded above for the raycaster's
        # kelvin map). eos_t_amb_k stays 290 (ruling 6, a deliberate exception
        # to kelvin_ambient); S_EOS is the phi_exp*k_temp_to_kelvin slope
        # mechanism, value-frozen to 1.0 exactly this arc (byte-identical).
        self.eos.T_AMB_K        = float(_ts.eos_t_amb_k)
        self.eos.C              = float(_ts.C)
        self.eos.S_EOS          = float(_ts.eos_slope)
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
        # n_work_ref (energy-books arc, design §2.4, RULING 2026-08-17): the
        # compression-work trust gate's reference density. PLUMBING ONLY at
        # P-E2b — the fade mechanism itself is P-E4's; this bind exists so the
        # dial is reachable from config ahead of that patch, and is provably
        # inert (nothing downstream reads self.eos.n_work_ref yet).
        self.eos.n_work_ref      = _ep("n_work_ref", self.eos.n_work_ref)
        # P-E3 (energy-books arc, design §2.8, NEW patch): interior momentum
        # drag with a heat counterparty. k_drag default 0.0 -> the mechanism
        # ships SILENT (dormancy BY BRANCH on the quantized fold, not this
        # float — see eos_solver.cpp). k_drag_heat_frac default 1.0 (RULING
        # R2, Erik 2026-08-17) keeps the conservation oracle EXACT through
        # every gate; Erik sweeps the fraction at P-E5.
        self.eos.k_drag          = _ep("k_drag", self.eos.k_drag)
        self.eos.k_drag_heat_frac = _ep("k_drag_heat_frac",
                                        self.eos.k_drag_heat_frac)
        # c_v: EOSSolver's own copy of the SAME [physics.thermal] c_v gas
        # heat-capacity constant self.temperature.c_v was bound from above —
        # ONE config key, two solvers, the n_floor_heat/T_MAX_PHYS precedent.
        self.eos.c_v              = float(getattr(thermal, "c_v", 1.0))

        # P-E3 forbidden-band load-warn tripwire (design §2.8, task item D):
        # the storm audit's §5 window (docs/storm_audit_2026-08-14.md §5 row
        # d/d', §5B) is dormant at shipped dials (k_wind_strip=0.0) but must
        # never be wandered into unwarned. [materials.air] wave_absorb in the
        # OPEN interval (0, 0.02) WHILE k_wind_strip > 0 is that historical
        # rectifier window — warn loudly at load, naming the audit, rather
        # than silently let a config drift back into it.
        air_cfg = getattr(CFG.materials, "air", None)
        air_wave_absorb = float(getattr(air_cfg, "wave_absorb", 0.0)) \
            if air_cfg is not None else 0.0
        if 0.0 < air_wave_absorb < 0.02 and self.fire.params.k_wind_strip > 0.0:
            warnings.warn(
                f"[materials.air] wave_absorb={air_wave_absorb} sits inside "
                "the FORBIDDEN BAND (0, 0.02) while [physics.fire] "
                f"k_wind_strip={self.fire.params.k_wind_strip} > 0 — this is "
                "the storm audit's §5 instability window (docs/storm_audit_"
                "2026-08-14.md §5 mechanism, row d/d', §5B): damp "
                "0.002-0.01 + a live wind-strip fire is the measured "
                "violently-unstable regime (KE burst, T-floor spiral to the "
                "T_MIN rail). Raise wave_absorb to >= 0.02, use k_drag "
                "instead ([physics.eos] k_drag, design §2.8 — a strictly "
                "stronger sink at the equivalent rate, A@0.02 == k_drag "
                "0.0067), or set k_wind_strip = 0.0.",
                RuntimeWarning, stacklevel=2)

        # P-T0 (energy-books arc, 2026-08-17, docs/energy_transport_design_
        # 2026-08-16.md §2.6 — the trace 0% ruling): `trace_mass_scale`
        # retired from EOSSolver entirely (the C++ member is gone, not
        # wired to 0.0). LOUD guard, the P-S1 `smoke_emission` idiom: a
        # config that still carries the key would otherwise silently do
        # nothing (this binding never read it even before P-T0 — the
        # struct's own default was always used), and whoever set it would
        # never find out why it has no effect.
        if getattr(eos_cfg, "trace_mass_scale", None) is not None:
            raise RuntimeError(
                "[physics.eos] still carries 'trace_mass_scale' — this key "
                "was retired at P-T0 (traces left the Dalton sum entirely; "
                "N_total is now exactly n_bulk). Remove it from "
                "config.toml. See docs/energy_transport_design_2026-08-16."
                "md §2.6 and docs/e1_p_t0_asbuilt_2026-08-17.md."
            )

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
        # demand = burn_rate*I*o2f*dt. o2_frac_ext/full are the SAME law the fire
        # logistic uses (bound from [physics.fire] so there is one source of
        # truth); o2_thresh_burn is now only an epsilon skip-floor. o2_frac_amb
        # is refreshed per-map in _ambient_args (the level's [ambient] o2_frac)
        # but is NOT read by the law since the full-response reference split.
        fire_cfg_c = getattr(CFG.physics, "fire", None)
        self.combustion.o2_frac_ext = float(
            getattr(fire_cfg_c, "o2_frac_ext", FIRE_O2_FRAC_EXT))
        self.combustion.o2_frac_full = float(
            getattr(fire_cfg_c, "o2_frac_full", FIRE_O2_FRAC_FULL))
        self.combustion.o2_frac_amb = float(
            getattr(fire_cfg_c, "o2_frac_amb", FIRE_O2_FRAC_AMB))
        self.combustion.o2_thresh_burn = _cp(
            "o2_thresh_burn", self.combustion.o2_thresh_burn)
        # --- o2_potency: THE SIZING RULING's preserved option ----------------
        # Erik's sizing ruling (2026-08-02) shipped PACKAGE A — draw_r = 2, NO
        # potency now — but ruled potency PRESERVED as an explicit option rather
        # than deleted. It is ONE config key, applied HERE, at load time, as a
        # multiplier on the baked heat-per-O2 constants (the H_fuel gas-side
        # yield and the H_bed fuel-bed deposit). Folding it at load means ZERO
        # runtime cost: the combustion pass never sees the key.
        #
        # DEFAULT 1.0 IS BYTE-NEUTRAL. Multiplication by 1.0 is an exact IEEE
        # identity, so a default config bakes bit-identical constants and no
        # digest or golden can move. That neutrality is the whole reason the
        # option can ship dormant.
        #
        # THE PRICE (documented at the config key too): potency extracts more
        # heat per unit of oxygen, so a sealed room's FIXED oxygen inventory
        # buys proportionally more fire — SEALED-ROOM SMOTHERING WEAKENS BY THE
        # SAME FACTOR. Smothering is a ships requirement, so raising this is a
        # real trade, not a tuning convenience.
        self._o2_potency = float(getattr(comb_cfg, "o2_potency", 1.0))
        if not (self._o2_potency > 0.0):
            raise ValueError(
                f"[physics.combustion] o2_potency = {self._o2_potency} must be "
                f"> 0 (it multiplies the heat-per-O2 constants; zero or negative "
                f"would mean a fire that consumes oxygen and yields no heat).")
        self.combustion.H_fuel = _cp("H_fuel", self.combustion.H_fuel) * self._o2_potency
        self.combustion.soot_yield = _cp("soot_yield", self.combustion.soot_yield)
        # v2.5 (P5.1 stoichiometric fuel consumption, design §5 v2.5 /
        # decisions #17): wall_hp consumed per unit N_O2 burned — THE
        # ember-lifetime dial. Quantized once per step in C++ like the
        # other per-step scalars.
        self.combustion.fuel_per_o2 = _cp(
            "fuel_per_o2", self.combustion.fuel_per_o2)
        self.combustion.o2_thresh_breathe = _cp(
            "o2_thresh_breathe", self.combustion.o2_thresh_breathe)
        # P-R4 (ruling A1): H_bed — the FUEL-BED deposit that owns the flame
        # plateau now the painter is gone. ONE logical constant split
        # mantissa/shift because the magnitude (order 1e5 T-counts per unit
        # N_O2) does not fit a Q16.16 mantissa; keeping the mantissa large is
        # also what keeps mul_q16's truncation fine against a per-tick burn of
        # only a few raw counts. Calibrated (like thermal_mass), NOT anchored.
        # o2_potency rides H_bed as well as H_fuel — the two together ARE the
        # heat-per-O2 chain, and scaling only one would tilt the gas-side /
        # fuel-bed split rather than the fire's power. The multiplier lands on
        # the MANTISSA (H_BED_SHIFT is a pure power of two and stays put), so
        # potency 1.0 is again an exact identity.
        self.combustion.H_BED_M = (
            _cp("H_BED_M", self.combustion.H_BED_M) * self._o2_potency)
        self.combustion.H_BED_SHIFT = int(
            getattr(comb_cfg, "H_BED_SHIFT", self.combustion.H_BED_SHIFT))
        # v2.4 rail: the SAME [physics.thermal].T_MAX_PHYS constant as the
        # thermal + EOS solvers (one ceiling in the system).
        self.combustion.T_MAX_PHYS = self._t_max_phys
        # P-O2b (design v5.2 "F-O2b") — THE EXTENDED OXYGEN DRAW. `draw_r` is
        # the BFS hop radius the burning tile draws oxygen over, expanded
        # through OPEN CELLS ONLY from its own open faces and weighted by
        # W_hop[d] * (the permeability-multiplicative path weight). It is
        # Erik's Option 2b: the entrainment stand-in that raises DELIVERY
        # without inflating room O2 inventories, so sealed-room smothering
        # stays exactly real. draw_r == 1 reproduces the pre-P-O2b 4-face law
        # BIT FOR BIT (the patch's regression oracle).
        self._draw_r = int(getattr(comb_cfg, "draw_r", 1))

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
    def step(self, gmap, sim_time, tick=0):
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
            return self._step_resident(gmap, sim_time, tick=tick)

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
        self.cast_fire_heat(gmap, tick=tick)

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
            # THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-07-30
            # .md §4 item 1): the EOS is the pass that ACTUALLY advects T in the
            # live engine, so the thermal medium has to reach it too. On a
            # thermal_solid tile (a crate) the solver now SKIPS both of its
            # `temperature` writes and treats the tile as an occluder in its T
            # backtrace — the TemperatureSolver owns an object's temperature.
            # `solid`/`dyn_permeability`/the cmask are untouched, so gas still
            # seeps through the crate at permeability 0.5 (shield, not seal).
            thermal_solid=gmap.thermal_solid,
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
            # COOL-SHIFT AXIS (2026-07-30): the per-tile ambient-decay shift
            # (`T -= T >> cool_shift[i]`), the LOSS-side twin of the
            # heat_inv_shift above. Per-material because the thermal-mass arc
            # made furniture a thermal solid whose ONLY loss channel is this
            # decay, and one global cannot be right for a hull plate and a
            # wooden crate at once.
            gmap.cool_shift,
            # FUEL-FRACTION AXIS (2026-07-30): the per-tile reciprocal of each
            # tile's material's OWN full-health hp, which the fire logistic's
            # fuel term F = clamp01(wall_hp/hp_full) reads. Per-material because
            # F means "the fraction of THIS tile's fuel left" and the retired
            # global [physics.fire] fuel_ref (60.0) is WOOD's hp — so a
            # full-health crate (hp 30) read F = 0.5 forever and could not clear
            # the sustain ceiling at ambient O2 at any intensity.
            gmap.fuel_recip,
            # PER-MATERIAL EXTINCTION TEMPERATURE (P-R3, 2026-07-31): the
            # per-tile Q16.16 foot of the fire logistic's `hot` ramp, derived
            # `ignition_temp[mat] - ignition_to_ext_delta`. Per-material because
            # `fire_T_ext` sits on the same axis as `ignition_temp`, and the
            # retired global (350) exceeded BOTH shipped ignition temps — a tile
            # could ignite below its own sustain floor and snap straight out.
            gmap.fire_T_ext_plane,
            gmap.gas, gmap.gases.conservative, self._o2_idx,
            sim_time,
            # BC: the ambient ring is wiped to ΔT=0 in the temperature pre-pass
            # (the vacuum-breach idiom); None on space maps = byte-identical.
            is_ambient=amb[0],
            # P-R4 (ruling A1.7): the SIGNED radiation accumulator the
            # fire-plane cast filled at the TOP of this same tick. The
            # temperature pass folds it FIRST in Pass 1 (before the heat
            # deposit), through each tile's own heat_inv_shift — so a tile's
            # radiative GAIN and its emitter's matching LOSS both convert this
            # tick, on one scale, with no painter in sight.
            rad_net=gmap.rad_net,
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
            # demand. o2_frac_ext/full are the SAME law the fire logistic uses.
            self.bp.cuda_combustion_step(
                gmap.gas, self._o2_idx, self._inert_n2_idx, self._black_smoke_idx,
                gmap.temperature, gmap.wall_hp, gmap.fire,
                gmap.flammable, gmap.solid, gmap.is_vacuum,
                ignition_temp_q16,
                sim_time, self.temperature.c_v, self.temperature.n_floor_heat,
                self.combustion.burn_rate, self.combustion.o2_thresh_burn,
                self.combustion.H_fuel, self.combustion.soot_yield,
                self.combustion.fuel_per_o2,
                self.combustion.o2_frac_ext, self.combustion.o2_frac_full,
                self.combustion.T_MAX_PHYS,
                # THERMAL-MASS AXIS, P-EOS (ruling §2 site 3) — see the CPU
                # branch below; the two backends must read the same masks.
                gmap.thermal_solid, gmap.heat_inv_shift,
                # P-R4 (ruling A1): the FUEL-BED deposit — the flame heating
                # its own fuel surface, which is what owns the flame plateau
                # now that the painter is retired. Same plane, same split
                # constant as the CPU branch below.
                gmap.heat,
                self.combustion.H_BED_M, self.combustion.H_BED_SHIFT,
                # D1: the error-feedback demand accumulator (synced, IN/OUT).
                gmap.dem_acc,
                # P-O2b: the extended draw. `max_claimants` is read off the
                # LIVE plane rather than re-read from config, so the depth the
                # C++ hard-check sees is by construction the depth that exists.
                self._draw_r,
                gmap.dyn_permeability,
                int(gmap.dem_acc.shape[0]),
            )
        else:
            self.combustion.step(
                gmap.gas, self._o2_idx, self._inert_n2_idx, self._black_smoke_idx,
                gmap.temperature, gmap.wall_hp, gmap.fire,
                gmap.flammable, gmap.solid, gmap.is_vacuum,
                ignition_temp_q16,
                sim_time, self.temperature.c_v, self.temperature.n_floor_heat,
                # THERMAL-MASS AXIS, P-EOS (docs/thermal_mass_eos_ruling_2026-
                # 07-30.md §2 site 3): a FURNITURE tile is an open, gas-holding
                # burn site but thermally an OBJECT, and under ruling A3 its pore
                # gas is thin — so the gas-divisor deposit would inflate the
                # object's T by ~2.5-3x per unit burn. On a thermal_solid burn
                # site the aggregate deposit converts through the tile's own
                # `heat_inv_shift` instead, exactly as a ray deposit does. Same
                # energy in, object-appropriate scale.
                gmap.thermal_solid, gmap.heat_inv_shift,
                # P-R4 (docs/radiation_raycaster_extinction_ruling_2026-07-31
                # .md A1): H_bed — the FUEL-BED deposit. With the painter gone
                # a lone crate's radiation nets to zero at the source and only
                # LOSES to cooler surroundings, so combustion has to own the
                # plateau. Each claimant gets H_bed * (the O2 it actually
                # consumed) as a positive, order-free add into `heat[]`, which
                # the temperature solver then converts through the tile's own
                # heat_inv_shift. Huggett-SHAPED, not Huggett-VALUED (see
                # combustion.h) — and it makes the plateau sag with local O2.
                gmap.heat,
                # D1 (ruling amendment 5): the error-feedback DEMAND
                # ACCUMULATOR. The per-claimant demand is ~1 Q16.16 count at
                # the operating point, and the old chained truncation floored
                # it to a staircase with a dead zone below I = 0.200 — a fire
                # born at ignition_seed 0.12 drew no oxygen and died. The
                # accumulator carries the wide product's sub-count remainder
                # across ticks, so the draw is exact in expectation and the
                # Huggett burn_rate anchor is untouched. Synced state, IN/OUT.
                gmap.dem_acc,
                # P-O2b (design v5.2 "F-O2b"): THE EXTENDED OXYGEN DRAW —
                # Erik's Option 2b. The burning tile draws O2 from every open
                # cell within `draw_r` BFS hops, reached through open cells only
                # from its own open faces, distance- and permeability-weighted;
                # the O2 is debited at those donors but the heat and soot land
                # at the FIRE (ruling 4, "air is heated at the fire only").
                # `dyn_permeability` is the plane the path weight rides —
                # crates attenuate, walls block. `max_claimants` is read off the
                # LIVE plane rather than re-read from config, so the depth the
                # C++ hard-check sees is by construction the depth that exists.
                self._draw_r,
                gmap.dyn_permeability,
                int(gmap.dem_acc.shape[0]),
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

    def _step_resident(self, gmap, sim_time, tick=0):
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
        self.cast_fire_heat(gmap, tick=tick)
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
        # THERMAL-MASS AXIS, P-EOS: `thermal_solid` JOINS this per-tick upload.
        # It was a static one-shot upload while only host code read it (P2's
        # recorded caveat); now DEVICE kernels read it (the resident SL advection
        # + compression work), and it is NOT static — `on_tile_changed` patches it
        # whenever a tile's material changes (a crate burning out) — so a
        # one-shot device copy would go stale exactly like is_ambient's did.
        gmap.from_host(["atmosphere", "wind_x", "wind_y", "temperature",
                        "gas", "solid", "is_vacuum", "is_ambient",
                        "dyn_permeability", "thermal_solid"])
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
            # THERMAL-MASS AXIS, P-EOS: the MIRROR (the host occlusion predicate,
            # like every other pre-stage reduction) + the DEVICE copy the SL and
            # compression kernels read. Uploaded fresh above.
            thermal_solid=gmap.thermal_solid,
            d_thermal_solid=dev["thermal_solid"],
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
            gmap.cool_shift,             # cool-shift axis (host mirror)
            gmap.fuel_recip,             # fuel-fraction axis (host mirror)
            gmap.fire_T_ext_plane,       # per-material T_ext (host mirror)
            gmap.gas, gmap.gases.conservative, self._o2_idx,
            sim_time,
            is_ambient=amb[0],
            # P-R4: the radiation accumulator rides the SAME host mirror the
            # rest of this bracket reads (the cast at step 1 filled it there).
            rad_net=gmap.rad_net,
        )
        # The mirror is now authoritative for every synced field (each stage wrote
        # it; the two resident loops' outputs were D2H'd to it). No final batched
        # D2H is needed — consumers read the mirror unchanged (the Q4 baseline);
        # the device set is re-uploaded whole next tick (from_host).
        return destroyed

    # ------------------------------------------------------------------
    # BC: planetside AMBIENT ring args (boundary_conditions_spec_2026-07-19)
    # ------------------------------------------------------------------
    @staticmethod
    def _ambient_mask(gmap):
        """The ``is_ambient`` argument every C++ ambient branch is gated on.

        ``None`` on a space map (no ambient config, or no ring tiles) — the
        null pointer is what makes the ambient path dormant BY BRANCH (spec
        §5); the live ring mask otherwise. Split out of ``_ambient_args``
        (P-M4b) so the SPACE-vs-AMBIENT decision exists exactly once and a
        read-only caller can reach it without ``_ambient_args``' per-tick side
        effects (the o2_frac_amb refresh, the n_amb cache fill).
        """
        amb = getattr(gmap, "_ambient", None)
        if amb is None or not gmap.is_ambient.any():
            return None
        return gmap.is_ambient

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
        if self._ambient_mask(gmap) is None:
            return (None, None, 0, None, None)
        # X_amb is this map's authored ambient O2 fraction — one source of truth
        # with the BC. Refresh both solvers each tick; o2_frac is static per map,
        # so this is a cheap idempotent set.
        # NOTE (full-response reference split, 2026-07-30): X_amb is NO LONGER the
        # mole fraction at which o2f saturates — that is o2_frac_full (pure O2),
        # which is deliberately NOT map-overridden. These two writes now only keep
        # the solvers' record of "what the ambient atmosphere is" current; neither
        # O2 law reads it. Enriching a map's [ambient] o2_frac therefore RAISES
        # o2f (more O2 in the air) instead of silently rescaling the law.
        self.fire.params.o2_frac_amb = float(amb.o2_frac)
        self.combustion.o2_frac_amb = float(amb.o2_frac)
        n_gases = gmap.gas.shape[0]
        if (self._ambient_n_amb is None
                or self._ambient_n_amb.shape[0] != n_gases):
            n_amb = np.zeros(n_gases, dtype=np.int32)
            n_amb[self._o2_idx] = int(amb.n_o2_q)
            n_amb[self._inert_n2_idx] = int(amb.n_n2_q)
            self._ambient_n_amb = n_amb
        return (self._ambient_mask(gmap), self._ambient_n_amb,
                int(amb.pin_q), gmap.sponge_sigma, gmap.sponge_udamp)

    # ------------------------------------------------------------------
    # P-M4b (mass-books arc): the energy books, readable from Python
    # ------------------------------------------------------------------
    def energy_books_sum(self, gmap) -> int:
        """S = Σ n_bulk·T over the energy books' accountable set (raw Q16.16²).

        THE instrument for the arc's energy-seam gates: bracket any state edit
        (a ``destroy_wall``, a weapon, a whole tick) with two calls and diff
        them.

        The skip-set — ``solid || thermal_solid || is_vacuum ||
        (ambient_mode && is_ambient)`` — and the ``Σ n_bulk·T`` arithmetic live
        in C++ (``eos_energy_books_sum``, eos_solver.cpp), which is the SAME
        routine ``EOSSolver::step``'s own ``eth_transport_delta`` /
        ``eth_compression_delta`` brackets call. Nothing here re-implements it:
        this method only ASSEMBLES the planes, and it assembles them from the
        very expressions ``step`` threads into ``run_substeps`` (``gmap.gas``,
        ``gmap.gases.conservative``, ``gmap.temperature``, ``gmap.solid``,
        ``gmap.is_vacuum``, ``_ambient_mask(gmap)``, ``gmap.thermal_solid``)
        — so a plane that changes over there changes here in the same edit.
        Do NOT inline the skip-set into Python: drift between the books and the
        instrument that reads them is the failure this arc is about.

        ``_ambient_mask`` is the ONE space-map-vs-ambient-map decision
        ``_ambient_args`` itself makes (``None`` -> the C++ ring term is dormant
        by branch), not a second copy of it.

        Pure instrumentation: reads state, writes none, folds into no digest.
        """
        return int(self.bp.eos_energy_books_sum(
            gmap.gas, gmap.gases.conservative, gmap.temperature,
            gmap.solid, gmap.is_vacuum,
            is_ambient=self._ambient_mask(gmap),
            thermal_solid=gmap.thermal_solid,
        ))

    # ------------------------------------------------------------------
    # K2: sim-side fire heat ray pass
    # ------------------------------------------------------------------
    def cast_fire_heat(self, gmap, tick=0):
        """Deposit fire's radiant heat into ``gmap.heat`` (proposal §1).

        Enumerate every burning tile (``fire > 0``) in fixed ROW-MAJOR order,
        turn each into a short-range heat source, and cast the whole list with
        the C++ raycaster into ``gmap.heat`` — Q16.16, saturating-add, occluded
        per tile by ``gmap.heat_atten`` (K1). HEAT-ONLY: the render light
        buffers are throwaway scratch (fire's visual glow is a separate later
        step) and ``smoke_glow`` is skipped (None).

        P-R1 (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1):
        the source build — enumerating ``fire > 0`` and turning each tile into
        a per-source ``(x, y, max_range, angle_center, intensity, heat)``
        tuple — now runs INSIDE the C++ raycaster
        (:meth:`Raycaster.cast_from_fire_plane` /
        ``cuda_raycaster_cast_from_fire_plane``), ONE call per tick, instead of
        a Python loop building one ``bp.LightSource()`` per burning tile
        (~10 pybind attribute writes each, ~6000/tick at 600 fires). The
        per-source parameters and the march itself are UNCHANGED — this is a
        mechanical relocation, gated byte-identical on ``heat`` against the
        pre-patch Python loop.

        Determinism (must hold — ``heat`` is sim-affecting and feeds ignition /
        unit damage downstream):

        - **Fixed ray count, fixed angles, NO RNG.** Each source uses exactly
          ``fire_ray_count`` (8) rays with ``jitter == 0``. The 8 rays are evenly
          spaced over the full circle by the C++ march; a fixed per-source phase
          (``angle_center``) derived from the tile coords rotates the fan so
          neighbouring fires don't all fire the same 8 directions — but it is a
          pure function of (row, col), never random. No ``sim.rng`` is touched.
        - **Fixed source order.** Row-major enumeration of the burning tiles
          (the C++ enumeration walks the fire plane in the same row-major order
          ``np.nonzero`` used to yield).
        - **Integer saturating-add deposit.** Order-independent -> bit-identical
          across machines / runs (the property that lets ``heat`` be a CUDA
          atomicAdd).

        Called at the START of :meth:`step`, BEFORE the TemperatureSolver.
        """
        # S3a: gmap.fire is int32 Q16.16. The `> 0` burning mask is exact on the
        # integer field (0 counts == unlit). The per-tile INTENSITY that feeds
        # the heat-ray range/intensity params is now dequantized INSIDE the C++
        # source build (P-R1) — this Python-side check only decides whether
        # there is anything to cast at all.
        fire = gmap.fire
        # Fast out: no EMITTERS -> nothing to exchange (rad_net stays whatever
        # it was; the sim clears it at end of tick). P-R4 (ruling A1.8) widened
        # the emitter set from `burning` to `burning ∪ (thermal_solid && T >=
        # T_emit_gate)`, so the dormancy test widens with it. The temperature
        # leg is a plain MAX reduction (no temporaries, no mask allocation): it
        # is a NECESSARY condition — if no tile anywhere is at the gate then no
        # warm emitter exists — and the C++ builder applies the exact per-tile
        # predicate. Dormant maps still cost two cheap reductions per tick.
        t_emit_q = int(round(float(self.raycaster.T_emit_gate) * 65536.0))
        if not bool((fire > 0).any()) and int(gmap.temperature.max()) < t_emit_q:
            return

        h, w = fire.shape
        # P-F1a: THE THROWAWAY LIGHT BUFFERS ARE GONE FROM THIS CALL.
        #
        # Until now the fire cast wrote RGB + direction into scratch planes that
        # this method then THREW AWAY — fire's visible glow is drawn by the
        # renderer's own blackbody selector (renderer/fire_lights.py), which
        # never read them. That was merely wasteful while light and radiation
        # shared ONE march. It stopped being merely wasteful when v7 rule 4 split
        # them: the EMISSION cast (long rays, pure-radiation fast path) and the
        # VISIBLE-LIGHT cast (short rays, legacy machinery) are two separate
        # marches now, and on CUDA two separate DEVICE ROUND-TRIPS — upload the
        # plane set, launch, download, twice over. Measured on this box that
        # second round-trip costs ~2.1 ms, which DWARFS the +0.095 ms the
        # >= grid-diagonal rays themselves add.
        #
        # So the runner passes None and the light cast is skipped entirely, on
        # both backends. BEHAVIOURALLY NEUTRAL — the buffers were discarded — and
        # it is exactly why the C++/binding API keeps the light cast as an
        # OPTION rather than deleting it: any caller that genuinely wants fire's
        # light still gets bit-for-bit what it always got.
        #
        # `_fire_scratch_rgb`/`_dx`/`_dy` stay declared (and permanently None) so
        # a stale external reference fails loudly rather than silently reading a
        # buffer nothing writes any more.

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
        # CUDA-S2 LIVE: pick the per-tick cast backend ONCE (a pure flag read;
        # constant False on the CPU build). P-R1: BOTH entry points enumerate
        # the SAME fire plane in the SAME C++ row-major order and build
        # byte-identical per-source params (Raycaster::build_fire_sources'
        # float-parity contract) — they differ only in which EXISTING march
        # machinery consumes the resulting source list: the CPU cast runs
        # cast_source_directional per source in place
        # (Raycaster::cast_from_fire_plane), the CUDA cast concatenates every
        # source's build_ray_list into ONE device march
        # (cuda_raycaster_cast_from_fire_plane, S8c's batched path) — so `heat`
        # is byte-for-byte the same either way (the S2/S8c gates proved the
        # march + batching; P-R1 only relocated the source build). The
        # light_rgb/dir buffers also round-trip to the host on the CUDA call
        # (render-only / deterministic-exempt).
        use_cuda_ray = bool(self._raycaster_on_cuda())
        if use_cuda_ray:
            bp.cuda_raycaster_cast_from_fire_plane(
                self.raycaster,
                fire,
                self.fire_ray_count,
                self.fire_range_base, self.fire_range_per_i,
                self.fire_intensity_base, self.fire_intensity_per_i,
                self.fire_color,
                None,                 # light_rgb: discarded -> skip the light cast
                None,                 # light_dx
                None,                 # light_dy
                self._fire_gas_f,
                gmap.gases.absorption,
                gmap.gases.scatter_albedo,
                gmap.dyn_light_atten,
                gmap.heat_atten,      # a_x: absorptivity == emissivity (Kirchhoff)
                gmap.temperature,     # both ends' E° lookup
                gmap.heat_inv_shift,  # the limiter's per-end budget
                gmap.thermal_solid,   # the warm-emitter mask
                gmap.rad_net,         # <- the SIGNED tile ledger
                gmap.rad_amb,         # <- rule 4: the per-tile SKY ledger
                gmap.rad_flux,        # <- D3: the damage SENSOR (not the ledger)
                int(tick),            # <- D4: the fan's per-tick phase rotation
                None,                 # smoke_glow: skipped (render-only, later)
            )
        else:
            self.raycaster.cast_from_fire_plane(
                fire,
                self.fire_ray_count,
                self.fire_range_base, self.fire_range_per_i,
                self.fire_intensity_base, self.fire_intensity_per_i,
                self.fire_color,
                None,                 # light_rgb: discarded -> skip the light cast
                None,                 # light_dx
                None,                 # light_dy
                # Multi-gas march (engine/05 §6.2): pass the full gas array +
                # per-gas tables. Gases NEVER attenuate the heat channel (only
                # material heat_atten does), so the radiation exchange — the
                # only output that survives this cast (smoke_glow=None) — is
                # bit-identical to the pre-multigas single-smoke call. S2b:
                # dequantized float bridge.
                self._fire_gas_f,
                gmap.gases.absorption,
                gmap.gases.scatter_albedo,
                gmap.dyn_light_atten,
                gmap.heat_atten,      # a_x: absorptivity == emissivity (Kirchhoff)
                gmap.temperature,     # both ends' E° lookup
                gmap.heat_inv_shift,  # the limiter's per-end budget
                gmap.thermal_solid,   # the warm-emitter mask
                gmap.rad_net,         # <- the SIGNED tile ledger
                gmap.rad_amb,         # <- rule 4: the per-tile SKY ledger
                gmap.rad_flux,        # <- D3: the damage SENSOR (not the ledger)
                int(tick),            # <- D4: the fan's per-tick phase rotation
                None,                 # smoke_glow: skipped (render-only, later)
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

    # (_step_ripple() DELETED - audit Patch A / A9, 2026-08-04. The ripple
    # pass moved into C++ (PhysicsEngine::step_tail, which reproduces this
    # method's dormancy guard - see physics_engine.h). Nothing called the
    # Python one; tests/test_water_ripple.py:304 already records that the old
    # monkeypatch no longer intercepts it.)
