"""Atmosphere mass conservation with partial (soft-unit) permeability.

Regression net for the DOOR-STAMP LEAK: `stamp_units` painted a living unit's
footprint into ``dyn_permeability`` at 0.5 with no solid clamp, so a unit
standing on a closed DOOR tile (passable to movement, solid to flow) raised
the door's permeability 0 -> 0.5. The implicit Gauss-Seidel diffusion in
``atmosphere_solver.cpp`` then opened a face into a cell it SKIPS (walls are
not unknowns) and HARD-ZEROES each substep — a one-sided Dirichlet p=0 flux
with conductance 0.5 that destroyed mass. With the 7 level units spawned, a
sealed airtight ship drained at a few %/s down to near-vacuum.

Two-layer fix pinned here:

1. C++ (``atmosphere_solver.cpp`` GS): a face into an excluded+zeroed cell
   (obstacle/wall) contributes 0 to BOTH the numerator and the diagonal sum
   (proper Neumann), for ANY permeability the field carries. Bit-identical
   for perm in {0,1} (static walls already carry perm 0).
2. Python (``gamemap.stamp_units``): the unit stamp takes MIN with the static
   permeability — a body can make an open tile porous, never make a sealed
   tile leak.

The solver tests below build the scene from raw arrays (no level load — the
house pattern of the synthetic solver tests); the stamp test uses the real
GameMap + a door tile from unhcr_vessel.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_atmosphere_conservation.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_DOOR  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402
from simulation.unit import Unit  # noqa: E402

H = W = 24          # sealed box
TICKS = 100
PATCH = 3           # unit-footprint-sized perm patch
UNIT_PERM = 0.5
REL_TOL = 1e-3      # conservation tolerance (tiny unconverged-GS float drift
                    # is acceptable; the bug was percent-per-second scale)


# ---------------------------------------------------------------------------
# Raw sealed-box harness around the config-bound AtmosphereSolver (exactly the
# params the game runs, via PhysicsRunner — the test_fire_feedback pattern).
# ---------------------------------------------------------------------------
class _SealedBox:
    def __init__(self):
        solid = np.zeros((H, W), dtype=bool)
        solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
        self.solid = solid
        self.is_vacuum = np.zeros((H, W), dtype=bool)
        self.static_perm = np.where(solid, 0.0, 1.0).astype(np.float32)
        # Non-uniform initial field: a uniform field is a fixed point of the
        # GS for ANY face weights and cannot leak, so it would hide the bug.
        self.atmosphere = np.where(solid, 0.0, 1.0).astype(np.float32)
        self.atmosphere[5:8, 5:8] += 0.3
        self.wave_p = np.zeros((H, W), np.float32)
        self.wave_v = np.zeros((H, W), np.float32)
        self.wave_source = np.zeros((H, W), np.float32)
        self.wind_x = np.zeros((H, W), np.float32)
        self.wind_y = np.zeros((H, W), np.float32)
        self.absorb = np.zeros((H, W), np.float32)
        self.perm = self.static_perm.copy()

        self.solver = PhysicsRunner(bp).atmos     # config-bound params
        sim_time = 1.0 / float(CFG.clock.ticks_per_second)
        n = max(1, int(math.ceil(sim_time / self.solver.max_dt())))
        self.n_sub = n
        self.dt_sub = sim_time / n

        self.interior = (~solid) & (~self.is_vacuum)
        self.total0 = float(self.atmosphere[self.interior].sum())

    def stamp(self, y0, x0):
        """Rebuild perm = static, then paint a PATCH x PATCH block at
        (y0, x0) with UNIT_PERM — RAW, exactly like the pre-fix unit stamp
        (no solid clamp), so the solver-side guarantee is what is pinned."""
        self.perm[:] = self.static_perm
        y1 = min(y0 + PATCH, H)
        x1 = min(x0 + PATCH, W)
        self.perm[max(y0, 0):y1, max(x0, 0):x1] = UNIT_PERM

    def tick(self):
        for _ in range(self.n_sub):
            self.solver.step(
                self.wave_p, self.wave_v, self.wave_source, self.atmosphere,
                self.wind_x, self.wind_y,
                self.solid, self.solid, self.is_vacuum,
                self.perm, self.absorb, self.dt_sub,
            )

    def total(self):
        return float(self.atmosphere[self.interior].sum())


def test_moving_patch_in_open_air_conserves():
    """A 3x3 perm-0.5 patch (a walking soft unit) marched one tile per tick
    back and forth across the room: total interior mass conserved."""
    box = _SealedBox()
    span = list(range(2, W - PATCH - 1))
    path = span + span[::-1]
    for t in range(TICKS):
        box.stamp(11, path[t % len(path)])
        box.tick()
    drift = abs(box.total() - box.total0) / box.total0
    assert drift < REL_TOL, (
        f"moving open-air patch leaked mass: {box.total0:.4f} -> "
        f"{box.total():.4f} (rel drift {drift:.2e})"
    )


def test_patch_overlapping_wall_conserves():
    """THE bug config: the patch path hugs the top wall so its first row
    stamps perm 0.5 onto SOLID ring cells every tick (a unit walking over
    closed door tiles). The solver must treat faces into excluded+zeroed
    cells as sealed regardless of the permeability they carry.

    Pre-fix this destroyed ~40%+ of total mass in 100 ticks; post-fix it
    conserves to float-drift level."""
    box = _SealedBox()
    span = list(range(1, W - PATCH - 1))
    path = span + span[::-1]
    stamped_solid_seen = False
    for t in range(TICKS):
        x0 = path[t % len(path)]
        box.stamp(0, x0)                       # row 0 of the patch == wall ring
        stamped_solid_seen |= bool(
            (box.perm[box.solid] > 0.0).any())
        box.tick()
    assert stamped_solid_seen, "test setup broken: no solid cell was stamped"
    drift = abs(box.total() - box.total0) / box.total0
    assert drift < REL_TOL, (
        f"perm-stamped wall face leaked mass: {box.total0:.4f} -> "
        f"{box.total():.4f} (rel drift {drift:.2e})"
    )


def test_unit_stamp_never_raises_solid_permeability():
    """GameMap layer: a unit whose footprint covers a DOOR tile (passable to
    movement, solid to flow) must leave the door flow-sealed (perm 0) while
    its air tiles get the partial body permeability."""
    g = GameMap(load_level("unhcr_vessel"))
    doors = np.argwhere(g.material == MAT_DOOR)
    assert len(doors) > 0, "unhcr_vessel should contain door tiles"
    # Pick a door that is solid to flow (static perm 0) — the leak config.
    solid_doors = [tuple(d) for d in doors if g.solid[d[0], d[1]]]
    assert solid_doors, "expected at least one flow-solid door tile"
    dy, dx = solid_doors[0]

    # Centre the 3x3 footprint on the door: top-left = (door - 1).
    u = Unit("DoorStander", x=dx - 1, y=dy - 1, team=0)
    g.stamp_units([u])

    assert g.dyn_permeability[dy, dx] == 0.0, (
        "a unit standing on a solid door must not raise its permeability "
        f"(got {g.dyn_permeability[dy, dx]})"
    )
    expected = np.float32(getattr(CFG.physics, "unit_permeability", 0.5))
    for (tx, ty) in u.occupied_tiles():
        if not g.solid[ty, tx]:
            assert g.dyn_permeability[ty, tx] == expected, (
                f"air tile ({tx},{ty}) under the unit should carry the "
                f"partial body permeability {expected}"
            )
