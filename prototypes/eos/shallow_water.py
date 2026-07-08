"""
Faithful-but-minimal port of Breach's shipped water model, used ONLY by
scenario S5 to evolve `state.water_depth` as a flooding front.

This transcribes the SHAPE of docs/architecture/engine/07_fluid_and_water.md
§2 (the "pipe + damped velocity" model -- note the engine docs explicitly
reject the full shallow-water equations, §4 of that chapter; this file's name
matches the P0 task brief's shorthand, not the discarded scheme):

    surface        = floor_height + tilt + water_depth
    flow_velocity += dt * (-g * grad(surface) - damping * flow_velocity)
    water_depth   -= dt * div(flow_velocity * water_depth)   # upwind flux

Reflective walls: velocity zeroed on solid tiles, flux zeroed across any face
touching a solid tile, depth clamped >= 0 and zeroed on solid tiles. This is
a driver for S5, not the subject of the prototype -- it is NOT the real
`breach_physics` .pyd.
"""

import numpy as np

from state import State, CEILING_H

G = 9.81            # m/s^2
DAMPING = 3.0        # 1/s, linear velocity damping (settles into a flat pool)
SUBSTEPS = 2         # engine/07 §3: "unconditionally stable, run one or two steps per frame"


def _grad_with_walls(field: np.ndarray, solid: np.ndarray, dx: float):
    """Central-difference gradient, Neumann (mirrored) at solid neighbors --
    same wall-mirror stencil convention as the rest of this repo's prototypes
    (see prototypes/smoke_sim.py:compute_laplacian_with_walls). Because the
    outermost ring of every scenario's grid is always solid (see the MARGIN
    convention in scenarios.py), this also correctly mirrors at the domain
    boundary despite using `np.roll` (no separate edge case needed)."""
    left, right = np.roll(field, 1, axis=1), np.roll(field, -1, axis=1)
    up, down = np.roll(field, 1, axis=0), np.roll(field, -1, axis=0)

    wall_left, wall_right = np.roll(solid, 1, axis=1), np.roll(solid, -1, axis=1)
    wall_up, wall_down = np.roll(solid, 1, axis=0), np.roll(solid, -1, axis=0)

    left = np.where(wall_left, field, left)
    right = np.where(wall_right, field, right)
    up = np.where(wall_up, field, up)
    down = np.where(wall_down, field, down)

    gx = (right - left) / (2.0 * dx)
    gy = (down - up) / (2.0 * dx)
    return gx.astype(np.float32), gy.astype(np.float32)


class ShallowWaterDriver:
    """Owns the water velocity auxiliary (`vx`, `vy`) -- momentum that carries
    between ticks but that nothing outside this driver needs to read, exactly
    as engine/07 §2.1 describes it ("a velocity auxiliary ... exactly as the
    atmosphere's wave_v does"). `State` only carries `water_depth`, the
    quantity render/gameplay/the W3 helper below actually read.

    Not a `Solver` (no `--scheme` selects this) -- a fixed driver, called
    directly by run.py only when the scenario is S5.
    """

    def __init__(self, state: State):
        self.vx = np.zeros(state.shape, dtype=np.float32)
        self.vy = np.zeros(state.shape, dtype=np.float32)

    def step(self, state: State, dt: float) -> None:
        dx = state.tile_size_m
        solid = state.solid
        sub_dt = dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            h = state.water_depth
            surface = (state.floor_height + state.tilt + h).astype(np.float32)

            gx, gy = _grad_with_walls(surface, solid, dx)
            self.vx += sub_dt * (-G * gx - DAMPING * self.vx)
            self.vy += sub_dt * (-G * gy - DAMPING * self.vy)
            self.vx[solid] = 0.0
            self.vy[solid] = 0.0

            # Upwind face flux to the "next" cell (right / down): the depth
            # carried across a face is taken from the cell the flow comes
            # FROM. Zeroed across any face touching a wall -- which, thanks
            # to the always-solid outer ring, also kills the spurious
            # np.roll wraparound face for free.
            v_right = 0.5 * (self.vx + np.roll(self.vx, -1, axis=1))
            h_right = np.where(v_right > 0.0, h, np.roll(h, -1, axis=1))
            f_right = v_right * h_right
            f_right[solid | np.roll(solid, -1, axis=1)] = 0.0

            v_down = 0.5 * (self.vy + np.roll(self.vy, -1, axis=0))
            h_down = np.where(v_down > 0.0, h, np.roll(h, -1, axis=0))
            f_down = v_down * h_down
            f_down[solid | np.roll(solid, -1, axis=0)] = 0.0

            div = ((f_right - np.roll(f_right, 1, axis=1)) / dx
                   + (f_down - np.roll(f_down, 1, axis=0)) / dx)

            h = h - sub_dt * div
            np.clip(h, 0.0, None, out=h)
            h[solid] = 0.0
            state.water_depth = h.astype(np.float32)


# --- W3 air-coupling helper (engine/07 §5.1) --------------------------------
#
# Water decides how much floor-to-ceiling air column is left; a later air
# solver reads free_h and turns Delta(free_h) into an isothermal pressure
# source (atmosphere[i] *= free_h_before/free_h_after, ratio capped). The
# scaffold only provides the numbers -- no solver consumes them yet.

FREE_H_EPS = 1e-3


def free_air_height(water_depth: np.ndarray, ceiling_h: float = CEILING_H) -> np.ndarray:
    """Per-tile remaining air column above the water surface."""
    return np.maximum(ceiling_h - water_depth, FREE_H_EPS)


class FreeHeightTracker:
    """Tracks free_h across ticks so a later air solver can read
    (free_h_before, free_h_after) for this tick and derive the volume-
    displacement pressure ratio. Call `update()` once per tick, right after
    the water step."""

    def __init__(self, state: State, ceiling_h: float = CEILING_H):
        self.ceiling_h = ceiling_h
        self._prev = free_air_height(state.water_depth, ceiling_h)

    def update(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        free_h_after = free_air_height(state.water_depth, self.ceiling_h)
        before, after = self._prev, free_h_after
        self._prev = free_h_after
        return before, after
