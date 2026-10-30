"""P5z: the P5 idle in 3-D at game scale. The explore hub is
`idle_sim.IdleOctopus`'s (decide -> reach -> touch -> limp -> stick down),
inherited; the aim is re-written for the 3-D root (a yaw about the body's z in
the body frame, the arm staying on the deck).

`seated=False` gives the FREE animal of the no-slide gate: no anchors at all
(none at the start, none at the settle), so nothing but the water and the deck
can take a force from it.

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from arm_bank3d import World
from crawl_sim3d import Octopus3D
from game_scale import Row
from idle_sim import IdleOctopus, IdleParams


class IdleOctopus3D(IdleOctopus, Octopus3D):
    def __init__(self, *args, idle: IdleParams | None = None, seated: bool = True, **kw):
        super().__init__(*args, idle=idle, **kw)       # IdleOctopus -> Octopus3D; anchors every arm (seated)
        self.seated = seated
        if not seated:
            for a in range(8):
                self.bank.release(a)

    def _aim(self, a: int, ang: float) -> None:
        u0 = self.root_dir0[a] / np.linalg.norm(self.root_dir0[a])
        c, s = np.cos(ang), np.sin(ang)
        u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1], u0[2]])
        self.root_local[a, 1] = self.root_local[a, 0] + np.linalg.norm(self.root_dir0[a]) * u

    def _drive_arms(self, dt: float) -> None:
        super()._drive_arms(dt)
        if not self.seated:
            self.bank.anchored[:] = False              # the settle re-anchors in the 2-D hub: not for a free animal

    def post_tick(self) -> None:
        Octopus3D.post_tick(self)
        self.log_ex.append(list(self.ex))


def make_idle_octopus3d(row: Row, seed: int = 5, world: World | None = None, seated: bool = True,
                        mu: float | None = None, x0=(0.0, 0.0)) -> IdleOctopus3D:
    crawl = replace(row.crawl, seed=seed, p_recruit=0.0)   # no travel: the crawl hub never recruits
    return IdleOctopus3D(row.geom, row.chain, crawl, row.body, world=world, idle=row.idle, seated=seated, mu=mu, x0=x0)
