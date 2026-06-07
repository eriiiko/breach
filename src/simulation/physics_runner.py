"""PhysicsRunner — per-tick wrapper around the C++ ``breach_physics`` solvers.

Lifted from ``main.py`` (the inline ``PhysicsRunner`` shim from before
this migration) and reconciled with the parameter-binding side channel
in ``game.py:Physics._init_solvers`` (lines 754-800). The legacy code
set FireSimulation parameters (FIRE_D, FIRE_O2_THRESHOLD, etc.) and an
``max_source_per_step`` cap that ``main.py``'s PhysicsRunner silently
omitted — bringing both over here keeps fire / shockwave behavior
identical to the legacy entry point.

Fire parameter defaults match the legacy ``Physics.FIRE_*`` class
constants. They live on this module instead of in config.toml — that's
a known issue (architecture.md §6.5 flags it as TODO: move to config).
The migration intentionally does not change behavior; the config move
is a separate patch.

``step(gmap, sim_time)`` advances all physics by ``sim_time`` seconds
(usually one game tick = 1 / CFG.clock.ticks_per_second). The IMEX
substep loop is unchanged from the legacy. Returns the list of
``(y, x)`` coordinates where fire burned through walls — caller is
responsible for calling ``gmap.destroy_wall(y, x)`` on each.
"""
from __future__ import annotations

import math

from config import CFG


# ---------------------------------------------------------------------------
# Fire parameter defaults (lifted from game.py:Physics.FIRE_* class constants)
# ---------------------------------------------------------------------------
FIRE_D              = 0.3   # fire spread rate to neighbors
FIRE_O2_THRESHOLD   = 0.60  # fire dies below this atmosphere
FIRE_O2_CONSUMPTION = 0.3   # atmosphere consumed per step by fire
FIRE_SMOKE_EMISSION = 0.8   # smoke produced per step by fire
FIRE_WALL_DAMAGE    = 0.4   # HP damage to wall per step while burning
FIRE_K_WIND_THRESH  = 0.5   # fire must exceed this * wind_speed to survive
FIRE_K_WIND_NET     = 3.0   # rate of wind effect (both feeding and cooling)


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

        # AtmosphereSolver: wave + IMEX diffusion. Same params as legacy
        # _init_solvers; main.py's shim was missing max_source_per_step.
        self.atmos = bp.AtmosphereSolver()
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
        self.smoke = bp.SmokeDynamics()
        self.smoke.d_smoke              = float(CFG.physics.d_smoke)
        self.smoke.advection_rate       = float(CFG.physics.advection_rate)
        self.smoke.dt_scale             = float(CFG.physics.smoke_dt_scale)
        self.smoke.wind_diffusion_scale = float(CFG.physics.wind_diffusion_scale)

        # FireSimulation — the binding the legacy entry point did and
        # main.py's previous shim DID NOT. Without these the C++
        # defaults (set in FireParams in fire_simulation.h) take over;
        # they happen to match the constants here in the current build,
        # but the legacy intent was for the Python-side values to win.
        self.fire = bp.FireSimulation()
        self.fire.params.spread_rate    = FIRE_D
        self.fire.params.o2_threshold   = FIRE_O2_THRESHOLD
        self.fire.params.o2_consumption = FIRE_O2_CONSUMPTION
        self.fire.params.smoke_emission = FIRE_SMOKE_EMISSION
        self.fire.params.wall_damage    = FIRE_WALL_DAMAGE
        self.fire.params.k_wind_thresh  = FIRE_K_WIND_THRESH
        self.fire.params.k_wind_net     = FIRE_K_WIND_NET

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
        dt = self.atmos.max_dt()
        n = max(1, int(math.ceil(sim_time / dt)))
        dt_actual = sim_time / n
        for _ in range(n):
            self.atmos.step(
                gmap.wave_p, gmap.wave_v, gmap.wave_source, gmap.atmosphere,
                gmap.wind_x, gmap.wind_y,
                gmap.obstacles, gmap.solid, gmap.is_vacuum,
                gmap.dyn_permeability,
                gmap.dyn_wave_absorb,
                dt_actual,
            )
            self.smoke.step(
                gmap.smoke, gmap.wind_x, gmap.wind_y,
                gmap.obstacles, gmap.solid, gmap.is_vacuum,
                gmap.dyn_permeability,
                dt_actual * self.smoke.dt_scale,
            )

        destroyed = self.fire.step(
            gmap.fire, gmap.atmosphere, gmap.smoke, gmap.wall_hp,
            gmap.solid, gmap.flammable,
            sim_time,
        )
        return destroyed
