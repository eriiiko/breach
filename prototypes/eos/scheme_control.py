"""
Control scheme for the EOS Phase-1.2 visual prototype (P-ctrl patch).

Plain semi-Lagrangian Stable Fluids (Stam 1999): incompressible, constant
density. This is the non-EOS baseline both rung A (Feldman-O'Brien) and rung
B (Kwatra) must visually beat -- deliberately textbook and a touch diffusive
(PLAN.md's numerical diffusion comes for free from the bilinear interpolation
in semi-Lagrangian advection; no explicit extra diffuse pass is added here).
Does not touch N / EOS / eos_core. Unlike `PlaceholderSolver`, `state.vx`/
`state.vy` here ARE the real, persisted velocity field -- self-advected and
made divergence-free every tick -- not a decaying event residual layered
under a synthetic swirl.

Per-tick pipeline (PLAN.md's control spec):
  1. Forces -- see the note in `step()`: detonate/ignite impulses are staged
     into state.vx/vy (plus T/smoke) by scenarios.apply_event *before*
     solver.step() runs each tick, so there is nothing further to add here.
  2. Self-advect vx, vy (semi-Lagrangian).
  3. Project to divergence-free: fixed-count red-black Gauss-Seidel Poisson
     solve for laplacian(p) = div(u); Neumann (mirror) at solid, p=0 (never
     updated, so it just stays at its zero-initialized value) at vacuum;
     then u -= grad(p).
  4. Advect smoke, T on the projected velocity.
  5. Zero velocity in solid; drain smoke to 0 at vacuum (and solid).
"""

import numpy as np

from state import State, AMBIENT_T
from solver import Solver, _advect_semilagrangian


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
    """Central-difference divergence of (vx, vy). The caller has already
    zeroed vx/vy inside `solid` (the no-penetration wall condition -- a
    Dirichlet BC on velocity, distinct from the Neumann/mirror BC used for
    pressure below); the domain's outer ring is always solid too
    (scenarios.py), so the periodic np.roll wraparound never mixes in live
    fluid from the far edge."""
    ddx = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * dx)
    ddy = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * dx)
    return (ddx + ddy).astype(np.float32)


def _solve_pressure(rhs: np.ndarray, solid: np.ndarray, vacuum: np.ndarray,
                     sweeps: int) -> np.ndarray:
    """Fixed-count (NOT tolerance-based) red-black Gauss-Seidel relaxation
    for laplacian(p) = rhs. Red-black (checkerboard) ordering makes this a
    true Gauss-Seidel update -- each color's pass only ever reads neighbors
    of the *opposite* color, which the previous pass just refreshed -- while
    staying fully vectorized; one outer iteration (red pass + black pass)
    updates every fluid cell exactly once, i.e. is one "sweep".

    Neumann (mirror) at `solid`: mirroring a wall neighbor's value to the
    center's own value is algebraically equivalent, at the fixed point, to
    dropping that neighbor and using a reduced diagonal (the standard
    ghost-cell trick for a zero-gradient BC; ditto solver.py's `_diffuse`).

    Dirichlet p=0 at `vacuum`: vacuum cells are excluded from the update
    mask, so they stay at their zero-initialized value for the whole solve.
    Non-solid neighbor lookups use a plain (unmirrored) `np.roll`, so a
    fluid cell next to vacuum reads that true zero -- which is what creates
    the pressure gradient driving outrush into a breach.
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


class ControlSolver(Solver):
    """Plain semi-Lagrangian Stable-Fluids baseline -- see module docstring.
    Textbook and a touch diffusive by construction: the point is to give
    rung A / rung B something honest to visually beat."""

    name = "control"

    PRESSURE_SWEEPS = 40   # fixed count, not a convergence tolerance

    def step(self, state: State, dt: float) -> None:
        dx = state.tile_size_m
        solid = state.solid
        vacuum = state.vacuum

        # 1. Forces. A scheduled detonation/ignition is applied by
        # scenarios.apply_event, called from run.py once per scheduled
        # event -- BEFORE solver.step() runs that tick -- and writes its
        # radial impulse directly into state.vx/state.vy (plus state.T /
        # state.smoke). That write IS the event flag/field the scaffold
        # exposes to solvers: solver-agnostic, staged ahead of step(), per
        # scenarios.py's module docstring. So by the time this method runs,
        # this tick's kick (if any) is already sitting in state.vx/vy, and
        # step 1 is simply to treat them below as the live, real velocity
        # field -- nothing further to inject here.

        # 2. Self-advect velocity (semi-Lagrangian).
        vx1 = _advect_semilagrangian(state.vx, state.vx, state.vy, dt, dx)
        vy1 = _advect_semilagrangian(state.vy, state.vx, state.vy, dt, dx)
        vx1[solid] = 0.0
        vy1[solid] = 0.0

        # 3. Project to divergence-free.
        rhs = _divergence(vx1, vy1, dx) * (dx * dx)
        p = _solve_pressure(rhs, solid, vacuum, self.PRESSURE_SWEEPS)
        vx2, vy2 = _subtract_gradient(vx1, vy1, p, solid, dx)
        vx2[solid] = 0.0
        vy2[solid] = 0.0

        # 4. Advect smoke, T on the projected (divergence-free) velocity.
        smoke = _advect_semilagrangian(state.smoke, vx2, vy2, dt, dx)
        T = _advect_semilagrangian(state.T, vx2, vy2, dt, dx)

        # 5. Zero velocity in solid (already done above); drain smoke at
        # vacuum (and solid, matching PlaceholderSolver's open_air
        # convention so walls never carry stale smoke); walls stay ambient.
        open_air = state.open_air
        smoke[~open_air] = 0.0
        np.clip(smoke, 0.0, None, out=smoke)
        T[solid] = AMBIENT_T

        state.vx, state.vy = vx2, vy2
        state.smoke, state.T = smoke, T
        self.last_substeps = 1
