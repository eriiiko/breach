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
        # Smoke-side sink-pull toward the nearest breach (ch.05 smoke v2). The
        # dial Erik wants: 0 disables it (sealed-room behaviour is then bit-
        # identical to the plain semi-Lagrangian advection). Default 2.0 clears
        # a breached room in ~a dozen ticks while leaving a sealed room untouched.
        self.smoke.sink_strength        = float(
            getattr(CFG.physics, 'smoke_sink_strength', 2.0))

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

        # TemperatureSolver (engine/06 §1–§2): turns the per-tick `heat` deposit
        # into the persistent `temperature` field on solids (§1 conversion), then
        # spreads it by CONDUCTION (§2). The conduction pass keys faces off the
        # NO_FACE sentinel; bind it from config so Python (the per-tile
        # face_shift bake) and C++ never disagree. Ambient cooling (§3) + unit
        # damage (§4) land in later passes.
        self.temperature = bp.TemperatureSolver()
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
        # Smoke sink-direction field toward the nearest breach (ch.05 smoke v2).
        # Fetched once per tick: it only changes on topology edits, and the
        # accessor rebuilds it lazily (gated by gmap._sink_dirty), so this is a
        # cheap array hand-back on every tick except the rare one after a breach.
        sink_x, sink_y = gmap.sink_fields()
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
                sink_x, sink_y,
                gmap.obstacles, gmap.solid, gmap.is_vacuum,
                gmap.dyn_permeability,
                dt_actual * self.smoke.dt_scale,
            )

        destroyed = self.fire.step(
            gmap.fire, gmap.atmosphere, gmap.smoke, gmap.wall_hp,
            gmap.solid, gmap.flammable,
            sim_time,
        )

        # Heat -> temperature conversion + CONDUCTION + AMBIENT COOLING
        # (engine/06 §1.2 + §2 + §3, proposal §6 steps 1–3). Pass 1 reads the
        # `heat` deposit NON-DESTRUCTIVELY and accumulates it (scaled by
        # 1/thermal_mass via the precomputed shift) onto `temperature` on SOLID
        # tiles. Pass 2 spreads the just-converted field one conduction
        # relaxation step (gather, double-buffered) along the harmonic-mean face
        # shifts in `face_shift`; kappa==0 air faces are NO_FACE -> air stays
        # bit-exactly 0. Pass 3 sheds it: T -= T >> shift toward ambient (0),
        # using the smaller vacuum shift where a 4-neighbour is space-facing
        # (is_vacuum / atmosphere < o2_vacuum_thresh) — the same fields the
        # atmosphere/smoke solvers read, so a breached wall cools fast through
        # the existing seam. With no heat sources wired into the sim tick yet,
        # `heat` is 0 and a 0 field conducts/cools to nothing, so the field stays
        # 0 (no behaviour change) — but the seam is now in place for fire/beams.
        # Unit damage (§4) becomes a further pass in step().
        self.temperature.step(
            gmap.temperature, gmap.heat, gmap.heat_inv_shift,
            gmap.face_shift, gmap.solid,
            gmap.is_vacuum, gmap.atmosphere,
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
