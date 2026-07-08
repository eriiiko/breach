"""
Grid + state for the EOS Phase-1.2 visual prototype (P0 scaffold).

Dense float32 numpy fields on a 2D tile grid. No tile objects, no
`breach_physics` import -- arrays only, matching the shape of the real
engine's `GameMap` (docs/architecture/engine/02_state_and_ownership.md)
but reduced to exactly the fields this spike needs.

Array shape convention: (height, width) i.e. (ny, nx) -- row = y, col = x,
same as every other prototype in this repo (smoke_sim.py, fluid_tilted_ship.py)
and the same origin/flooring convention as engine/01_grid_and_coordinates.md.
"""

import numpy as np

# Per-level constants (engine defaults; see docs/architecture/engine/01 §.. and 07 §5.1).
TILE_SIZE_M = 1.0 / 3.0   # metres per tile
AMBIENT_T = 290.0         # K, ambient gas temperature
CEILING_H = 2.5           # m, floor-to-ceiling air column (W3 air-coupling, engine/07 §5.1)


class State:
    """All per-tile fields for one scenario instance.

    Fields:
        smoke         (H,W) float32, density >= 0
        T             (H,W) float32, gas temperature K (ambient AMBIENT_T)
        vx, vy        (H,W) float32, gas velocity m/s
        P             (H,W) float32, derived pressure-ish scalar -- NOT stepped
                      by anything in P0. Allocated so rung A/B can write
                      P = C*N*T into it without changing this class; P0 leaves
                      it at zero (see run.py / PLAN.md for why N isn't tracked
                      yet -- that's a rung A/B concern, not the scaffold's).
        solid         (H,W) bool, currently-blocking tiles (static walls AND
                      any closed door). This is the one mask solvers should
                      treat as "flow cannot cross".
        vacuum        (H,W) bool, hard vacuum (space / breached exterior).
        door          (H,W) bool, structural marker of "this tile is a door",
                      independent of open/closed. Open/closed is read off
                      `solid` (closed door => solid=True); events flip
                      `solid` at door tiles, `door` itself never changes.
        floor_height  (H,W) float32, static terrain height (m), default 0.
        tilt          (H,W) float32, ship-tilt height offset (m), default 0.
        water_depth   (H,W) float32, standing liquid depth (m), default 0.
    """

    def __init__(self, width: int, height: int, tile_size_m: float = TILE_SIZE_M):
        self.width = width
        self.height = height
        self.tile_size_m = tile_size_m
        shape = (height, width)

        # gas
        self.smoke = np.zeros(shape, dtype=np.float32)
        self.T = np.full(shape, AMBIENT_T, dtype=np.float32)
        self.vx = np.zeros(shape, dtype=np.float32)
        self.vy = np.zeros(shape, dtype=np.float32)
        self.P = np.zeros(shape, dtype=np.float32)

        # masks
        self.solid = np.zeros(shape, dtype=bool)
        self.vacuum = np.zeros(shape, dtype=bool)
        self.door = np.zeros(shape, dtype=bool)

        # water
        self.floor_height = np.zeros(shape, dtype=np.float32)
        self.tilt = np.zeros(shape, dtype=np.float32)
        self.water_depth = np.zeros(shape, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def open_air(self) -> np.ndarray:
        """Tiles that are neither solid nor vacuum -- where gas actually lives."""
        return ~self.solid & ~self.vacuum
