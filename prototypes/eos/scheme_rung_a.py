"""
Rung A -- Feldman-O'Brien prescribed-divergence incompressible solver.

EOS Phase-1.2 visual prototype (PLAN.md; docs/eos_research_report.md §4
"Rung A"). No acoustics, no momentum-carrying compressibility: hot gas
(P = C*N*T at held N, eos_core.py) is made to expand by PRESCRIBING a
velocity-divergence target and running ONE incompressible pressure
projection per tick to realize it. That is Rung A's whole cost story --
report §3's hot-core "250 -> ~1-4 substeps" becomes exactly 1 here (step 6).
Production precedent: Autodesk Bifrost ships this scheme (report §0/§2).

Two independent sources feed the prescribed divergence (step 1 below):
  - thermal expansion: hot open-air gas wants to expand -- the fireball
    push, driven by however hot `state.T` already is when this runs.
  - water displacement (S5 only): rising water shrinks the free-air column
    above it, so the trapped air is squeezed out sideways -- modeled the
    same way, as a positive divergence, via shallow_water's W3 free-air-
    column helper. Identically zero on every other scenario, because
    free_h never changes while water_depth stays at 0 (S1-S4).

Detonate/ignite events (scenarios.py `_apply_detonate` / `_apply_ignite`)
write directly into state.T/smoke/vx/vy *before* run.py calls this solver's
step() for the same tick -- that direct radial vx/vy kick is treated here as
Rung A's stand-in for the report's separate `wave_p` blast-impulse channel,
deliberately kept OUT of div_target (see the class docstring).

vx, vy, smoke, and T are all advected together on the pre-projection
velocity (step 2); only vx, vy then receive the projection's divergence
correction (step 3) -- smoke/T are not re-advected afterward.
"""

import numpy as np

from state import State, AMBIENT_T
from solver import Solver, _advect_semilagrangian
from eos_core import ensure_N, derive_pressure
from shallow_water import FreeHeightTracker


def _mirror_neighbors(field: np.ndarray, solid: np.ndarray):
    """Up/down/left/right neighbors of `field`, mirrored (Neumann, zero-
    gradient) at `solid` -- the same wall-mirror convention as solver.py's
    `_diffuse` / prototypes/smoke_sim.py:compute_laplacian_with_walls."""
    up = np.where(np.roll(solid, 1, axis=0), field, np.roll(field, 1, axis=0))
    down = np.where(np.roll(solid, -1, axis=0), field, np.roll(field, -1, axis=0))
    left = np.where(np.roll(solid, 1, axis=1), field, np.roll(field, 1, axis=1))
    right = np.where(np.roll(solid, -1, axis=1), field, np.roll(field, -1, axis=1))
    return up, down, left, right


def _divergence(vx: np.ndarray, vy: np.ndarray, dx: float) -> np.ndarray:
    """Central-difference divergence of (vx, vy). No wall-mirroring needed:
    the caller has already zeroed vx/vy inside `solid` (the no-penetration
    wall condition -- a Dirichlet BC on velocity, distinct from the Neumann/
    mirror BC used for pressure below); the domain's outer ring is always
    solid too (scenarios.py), so the periodic np.roll wraparound never mixes
    in live fluid from the far edge."""
    ddx = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * dx)
    ddy = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * dx)
    return (ddx + ddy).astype(np.float32)


def _solve_pressure(rhs: np.ndarray, solid: np.ndarray, vacuum: np.ndarray,
                     sweeps: int) -> np.ndarray:
    """Fixed-count (NOT tolerance-based -- determinism, report risk #4)
    red-black Gauss-Seidel relaxation for laplacian(p) = rhs, `rhs` already
    scaled by dx^2 by the caller. Red-black (checkerboard) ordering makes
    this a true Gauss-Seidel update -- each color's pass only ever reads
    neighbors of the *opposite* color, which the previous pass just
    refreshed -- while staying fully vectorized; one outer iteration (red
    pass + black pass) updates every fluid cell exactly once, i.e. is one
    "sweep" (~40, per PLAN.md / the report).

    Neumann (mirror) at `solid`: mirroring a wall neighbor's value to the
    center's own value is algebraically equivalent, at the fixed point, to
    dropping that neighbor and using a reduced diagonal (the standard
    ghost-cell trick for a zero-gradient BC; ditto solver.py's `_diffuse`).

    Dirichlet p=0 at `vacuum`: vacuum cells are excluded from the update
    mask, so they stay at their zero-initialized value for the whole solve
    (so a breach drains). Non-solid neighbor lookups use a plain
    (unmirrored) `np.roll`, so a fluid cell next to vacuum reads that true
    zero -- which is what creates the pressure gradient driving outrush.
    """
    p = np.zeros(rhs.shape, dtype=np.float32)

    wall_up = np.roll(solid, 1, axis=0)
    wall_down = np.roll(solid, -1, axis=0)
    wall_left = np.roll(solid, 1, axis=1)
    wall_right = np.roll(solid, -1, axis=1)

    yy, xx = np.mgrid[0:rhs.shape[0], 0:rhs.shape[1]]
    checker = (yy + xx) % 2 == 0
    fluid = ~solid & ~vacuum
    color_masks = (checker & fluid, ~checker & fluid)

    for _ in range(sweeps):
        for mask in color_masks:
            up = np.where(wall_up, p, np.roll(p, 1, axis=0))
            down = np.where(wall_down, p, np.roll(p, -1, axis=0))
            left = np.where(wall_left, p, np.roll(p, 1, axis=1))
            right = np.where(wall_right, p, np.roll(p, -1, axis=1))
            p = np.where(mask, (up + down + left + right - rhs) * 0.25, p)

    return p.astype(np.float32)


def _subtract_gradient(vx: np.ndarray, vy: np.ndarray, p: np.ndarray,
                        solid: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """u -= grad(p). Uses the same mirrored-at-solid neighbors as the solve
    itself, so no spurious force appears at a wall face; the actual no-
    penetration constraint is the hard vx/vy[solid] = 0 the caller applies."""
    up, down, left, right = _mirror_neighbors(p, solid)
    grad_x = (right - left) / (2.0 * dx)
    grad_y = (down - up) / (2.0 * dx)
    return (vx - grad_x).astype(np.float32), (vy - grad_y).astype(np.float32)


class RungASolver(Solver):
    """Feldman-O'Brien / Bifrost-style divergence-controlled incompressible
    gas solver -- see module docstring.

    Detonate/ignite events (scenarios.py `_apply_detonate` / `_apply_ignite`)
    write directly into state.T/smoke/vx/vy *before* run.py calls this
    solver's step() for the same tick. This solver deliberately does NOT
    also add a detonation pulse into div_target: the event's own direct
    radial vx/vy kick already stands in for the report's separate `wave_p`
    blast-impulse channel (report §2 "Rung A": "Feldman decouples fireball-
    expansion from blast-wave impulse by construction ... keeps its tuned
    wave_p as the separate blast-impulse channel"). This solver's own
    div_target contribution is purely the EOS/thermal-expansion term below,
    driven by whatever T the event already deposited -- exactly the
    "fireball on top of the impulse" split the report calls for.
    """

    name = "rungA"

    PRESSURE_SWEEPS = 40          # fixed count, not a convergence tolerance
    K_EXPAND = np.float32(0.25)   # 1/s, thermal-expansion divergence gain --
                                   # tuned WITH T_COOL so S4's ignite visibly
                                   # billows the smoke (~4 tiles over 60 ticks)
                                   # while velocities stay single-digit m/s and
                                   # fade as the fireball cools (self-verify).
                                   # S1's punchier ~12 m/s peak is the one-tick
                                   # detonation kick (the wave_p stand-in), NOT
                                   # this term -- it is unchanged by K_EXPAND.
    W_DISPLACE_GAIN = np.float32(60.0)
                                   # dimensionless amplifier on the W3 water-
                                   # displacement divergence. The physical
                                   # coupling is gain=1, but at the scaffold's
                                   # 2.5 m ceiling a 0.6 m flood trims the air
                                   # column only ~24%, and the front reaching
                                   # the smoke is shallow -- so the honest push
                                   # is ~0.003 m/s, invisible. This gain lifts
                                   # the advancing-front push to a spike-visible
                                   # nudge (smoke shoved by the flood) WITHOUT
                                   # changing the physical FORM of the term.
                                   # Purely an S5 visualization knob.
    T_COOL = np.float32(0.05)     # per-tick Newtonian relaxation of T toward
                                   # ambient. A fireball radiates its heat away
                                   # (report's one-way heat absorber; cf. the
                                   # placeholder's T_RELAX), so the prescribed
                                   # thermal expansion is a fading BLOOM, not a
                                   # permanent jet -- this is what keeps the
                                   # sustained expansion velocity gentle (a few
                                   # m/s) while still letting the initial bloom
                                   # be strong enough to visibly billow smoke.

    def __init__(self):
        super().__init__()
        self._free_h_tracker: FreeHeightTracker | None = None

    def step(self, state: State, dt: float) -> None:
        dx = state.tile_size_m
        solid = state.solid
        vacuum = state.vacuum
        air = state.open_air
        N = ensure_N(state)

        # ---- 1. prescribed divergence source -------------------------------
        div_target = np.zeros(state.shape, dtype=np.float32)

        # thermal expansion: hot gas wants to expand (the fireball push).
        excess = np.clip(state.T - AMBIENT_T, 0.0, None)
        div_target += np.where(air, self.K_EXPAND * excess / AMBIENT_T, 0.0)

        # water displacement (S5): rising water shrinks the free air column
        # -> the trapped air is squeezed out -> positive divergence.
        if self._free_h_tracker is None:
            self._free_h_tracker = FreeHeightTracker(state)
        free_h_before, free_h_after = self._free_h_tracker.update(state)
        water_div = -((free_h_after - free_h_before) / dt) / free_h_after
        div_target += np.where(air, self.W_DISPLACE_GAIN * water_div, 0.0)

        # ---- 2. semi-Lagrangian advection on the CURRENT (pre-projection)
        #         velocity -- vx, vy, smoke and T all ride the same field.
        vx0, vy0 = state.vx, state.vy
        vx1 = _advect_semilagrangian(vx0, vx0, vy0, dt, dx)
        vy1 = _advect_semilagrangian(vy0, vx0, vy0, dt, dx)
        smoke = _advect_semilagrangian(state.smoke, vx0, vy0, dt, dx)
        T = _advect_semilagrangian(state.T, vx0, vy0, dt, dx)
        vx1[solid] = 0.0
        vy1[solid] = 0.0

        # ---- 3. pressure projection with the prescribed divergence ---------
        #    solve laplacian(p) = div(u*) - div_target  (fixed-sweep RB-GS)
        #    then u -= grad(p)
        rhs = (_divergence(vx1, vy1, dx) - div_target) * (dx * dx)
        p = _solve_pressure(rhs, solid, vacuum, self.PRESSURE_SWEEPS)
        vx2, vy2 = _subtract_gradient(vx1, vy1, p, solid, dx)

        # ---- 4. thermal relaxation (fireball radiates -> expansion fades) ----
        T += (AMBIENT_T - T) * self.T_COOL

        # ---- 5. wall / vacuum cleanup ----------------------------------------
        vx2[solid] = 0.0
        vy2[solid] = 0.0
        smoke[~air] = 0.0
        np.clip(smoke, 0.0, None, out=smoke)
        T[solid] = AMBIENT_T
        N[vacuum] = 0.0   # in place; `N` already *is* state.N (ensure_N)

        state.vx, state.vy = vx2, vy2
        state.smoke, state.T = smoke, T

        # ---- 6. derive P = C*N*T ------------------------------------------------
        derive_pressure(state)

        # ---- 7. cost signal: one projection per tick, no acoustic substeps ------
        self.last_substeps = 1
