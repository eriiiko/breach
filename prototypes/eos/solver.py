"""
Pluggable solver interface for the EOS Phase-1.2 visual prototype.

`Solver` is the contract every fluid scheme (this file's `PlaceholderSolver`,
and the later rung A / rung B / control patches) implements. `run.py` drives
whichever one `--scheme` selects through exactly this interface, and the
timing harness (timing.py) reads `last_substeps` off it after every tick.

`PlaceholderSolver` is NOT a candidate scheme -- it exists only so the P0
scaffold runs end-to-end and produces a non-trivial GIF before rung A/B/control
land. It does something cheap and visible: a mild constant swirl blended with
whatever velocity events injected, semi-Lagrangian advection of smoke and T
along it, and a little diffusion.
"""

from abc import ABC, abstractmethod

import numpy as np

from state import State, AMBIENT_T


class Solver(ABC):
    """Base class for every fluid scheme.

    Subclasses mutate `state` in place each tick and must set
    `self.last_substeps` to how many internal substeps they took, so the
    timing harness can report substeps/tick alongside ms/tick -- the actual
    point of this spike (PLAN.md: "is rung B cheap enough to run in realtime?").
    """

    name: str = "solver"

    def __init__(self):
        self.last_substeps: int = 1

    @abstractmethod
    def step(self, state: State, dt: float) -> None:
        """Advance `state` in place by one tick of `dt` seconds.

        Must set `self.last_substeps` before returning.
        """
        raise NotImplementedError


def _advect_semilagrangian(field: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                            dt: float, dx: float) -> np.ndarray:
    """Backtrace each cell along (vx, vy) and bilinearly sample `field` there.
    Standard semi-Lagrangian advection (Stable Fluids); out-of-range samples
    clamp to the domain edge rather than wrap."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    src_x = np.clip(xx - vx * dt / dx, 0.0, w - 1.001)
    src_y = np.clip(yy - vy * dt / dx, 0.0, h - 1.001)

    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    fx = (src_x - x0).astype(np.float32)
    fy = (src_y - y0).astype(np.float32)

    top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
    bot = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def _diffuse(field: np.ndarray, solid: np.ndarray, amount: float) -> np.ndarray:
    """One Jacobi-style diffusion pass with Neumann (mirrored) walls -- same
    wall-mirror stencil convention used throughout this repo's prototypes
    (see prototypes/smoke_sim.py:compute_laplacian_with_walls)."""
    up, down = np.roll(field, 1, axis=0), np.roll(field, -1, axis=0)
    left, right = np.roll(field, 1, axis=1), np.roll(field, -1, axis=1)

    wall_up, wall_down = np.roll(solid, 1, axis=0), np.roll(solid, -1, axis=0)
    wall_left, wall_right = np.roll(solid, 1, axis=1), np.roll(solid, -1, axis=1)

    up = np.where(wall_up, field, up)
    down = np.where(wall_down, field, down)
    left = np.where(wall_left, field, left)
    right = np.where(wall_right, field, right)

    lap = up + down + left + right - 4.0 * field
    return (field + amount * lap).astype(np.float32)


class PlaceholderSolver(Solver):
    """Cheap, visible stand-in for the real fluid solvers. Not a candidate
    scheme -- just exercises the pipeline (see module docstring)."""

    name = "placeholder"

    SWIRL_SPEED = 1.2          # m/s, background swirl magnitude
    VELOCITY_DAMPING = 0.90    # per-tick decay of event-injected velocity
    DIFFUSION = 0.06
    SMOKE_DECAY = 0.995
    T_RELAX = 0.02             # per-tick relaxation toward ambient

    def __init__(self):
        super().__init__()
        self._swirl_vx: np.ndarray | None = None
        self._swirl_vy: np.ndarray | None = None

    def _swirl(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        """A mild constant rotational field about the grid centre, cached
        per grid shape. Pure cosmetics -- not a physical wind."""
        if self._swirl_vx is None or self._swirl_vx.shape != state.shape:
            yy, xx = np.mgrid[0:state.height, 0:state.width].astype(np.float32)
            cx, cy = state.width / 2.0, state.height / 2.0
            dx_, dy_ = xx - cx, yy - cy
            r = np.sqrt(dx_ * dx_ + dy_ * dy_) + 1e-3
            self._swirl_vx = (-dy_ / r * self.SWIRL_SPEED).astype(np.float32)
            self._swirl_vy = (dx_ / r * self.SWIRL_SPEED).astype(np.float32)
        return self._swirl_vx, self._swirl_vy

    def step(self, state: State, dt: float) -> None:
        swirl_vx, swirl_vy = self._swirl(state)

        # `state.vx/vy` hold ONLY the decaying event-injected residual
        # (detonate/ignite write their kick there before this runs). The
        # swirl is layered in fresh, every tick, purely as a local variable
        # for advection -- it must NEVER be written back into state.vx/vy,
        # or re-adding it on top of an already-swirl-inclusive field
        # compounds geometrically tick over tick (verified: this blew up to
        # ~19x SWIRL_SPEED within 20 ticks and advected all the smoke clean
        # off the grid before the fix).
        state.vx *= self.VELOCITY_DAMPING
        state.vy *= self.VELOCITY_DAMPING
        air = state.open_air
        adv_vx = np.where(air, state.vx + swirl_vx, 0.0).astype(np.float32)
        adv_vy = np.where(air, state.vy + swirl_vy, 0.0).astype(np.float32)

        dx = state.tile_size_m
        smoke = _advect_semilagrangian(state.smoke, adv_vx, adv_vy, dt, dx)
        T = _advect_semilagrangian(state.T, adv_vx, adv_vy, dt, dx)

        smoke = _diffuse(smoke, state.solid, self.DIFFUSION) * self.SMOKE_DECAY
        T = _diffuse(T, state.solid, self.DIFFUSION)
        T += (AMBIENT_T - T) * self.T_RELAX

        smoke[~air] = 0.0
        np.clip(smoke, 0.0, None, out=smoke)
        T[state.solid] = AMBIENT_T

        state.smoke, state.T = smoke, T
        # state.vx/vy already hold the damped residual (set above) -- left
        # as-is, deliberately excluding the swirl from both persisted state
        # and the rendered arrows (see note above).
        self.last_substeps = 1
