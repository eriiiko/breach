"""P5z: the P4 reach -> grab -> hold -> carry in 3-D at game scale, now with
the LIFT. The state machine is `grab_sim.GrabOctopus._drive_grab`, inherited;
what is 3-D here: the prop is a bank body with height (resting on the deck,
under g_eff, with the deck's friction), contact is a 3-D distance, the seal is
a 3-D rigid attachment in the prop's frame, the root aim keeps the 2-D yaw
and, from HOLD on, PITCHES the holding arm's root up by `lift_pitch_deg` over
`lift_s` -- the stiff proximal arm (hold activation 1.0, pulled in to 0.7)
then carries the prop off the deck. The crawl and the jets of the carry are
the P2 hub's and the P4 gate's.

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from arm_bank3d import BodyView, World
from crawl_sim3d import Octopus3D
from game_scale import GrabParams3D, Row
from grab_sim import CARRY, GRAB, HOLD, GrabOctopus


class GrabOctopus3D(GrabOctopus, Octopus3D):
    def __init__(self, *args, grab: GrabParams3D | None = None, **kw):
        super().__init__(*args, grab=grab, **kw)      # GrabOctopus -> Octopus3D; builds a 2-D prop we replace
        g = self.gp
        bank = self.world.bank
        bx = self.body.x3.copy()
        R = self.body.rot3()
        at = R @ np.array([g.prop_at[0], g.prop_at[1], 0.0])
        i = bank.add_body((bx[0] + at[0], bx[1] + at[1], 0.0), 0.0, mass=g.prop_mass, radius=g.prop_radius,
                          c_drag=1.0, area=2.0 * g.prop_radius * 0.05 * (g.prop_radius / 0.04),
                          rot_damp=self.bodyp.rot_damp, mu=self.crawl.mu_k)
        self.prop = BodyView(bank, i)
        self.root_dir0 = self.root_local[g.arm, 1] - self.root_local[g.arm, 0]
        self.aim_dir = self.root_dir0 / np.linalg.norm(self.root_dir0)
        self.t_hold = np.nan
        self.log_prop3 = []

    # -- 3-D helpers (the 2-D versions assumed planar vectors) ---------------------
    def _aim_root(self, a: int, target_world: np.ndarray) -> None:
        R = self.body.rot()
        root_w = self.body.x + self.root_local[a, 0, :2] @ R.T
        d = (np.asarray(target_world)[:2] - root_w) @ R
        d /= max(np.linalg.norm(d), 1e-12)
        u0 = self.root_dir0[:2] / np.linalg.norm(self.root_dir0[:2])
        ang = np.arctan2(u0[0] * d[1] - u0[1] * d[0], u0 @ d)
        ang = float(np.clip(ang, -np.radians(self.gp.aim_max_deg), np.radians(self.gp.aim_max_deg)))
        c, s = np.cos(ang), np.sin(ang)
        self.aim_dir = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1], 0.0])
        self._set_root_dir(a, 0.0)

    def _set_root_dir(self, a: int, pitch: float) -> None:
        u = np.array([np.cos(pitch) * self.aim_dir[0], np.cos(pitch) * self.aim_dir[1], np.sin(pitch)])
        self.root_local[a, 1] = self.root_local[a, 0] + np.linalg.norm(self.root_dir0) * u

    def _contact_nodes(self, a: int) -> np.ndarray:
        k0, k1 = self.gp.grab_nodes
        d = np.linalg.norm(self.bank.p[a, k0:k1 + 1] - self.prop.x3, axis=1)
        return np.arange(k0, k1 + 1)[d <= self.prop.radius + self.gp.sucker_reach]

    def _seal(self, a: int, nodes: np.ndarray) -> None:
        R = self.prop.rot3()
        for k in nodes:
            if any(g[0] == a and g[1] == k for g in self.bank.grabs):
                continue
            local = (self.bank.p[a, k] - self.prop.x3) @ R
            self.bank.add_grab(a, int(k), self.prop, local)

    def _wrap_side(self, a: int, k: int) -> float:
        p = self.bank.p[a]
        d = p[min(k + 1, len(p) - 1), :2] - p[max(k - 1, 0), :2]
        to_prop = self.prop.x - p[k, :2]
        return 1.0 if d[0] * to_prop[1] - d[1] * to_prop[0] > 0 else -1.0

    def _drive_grab(self, dt: float) -> None:
        phase_before = self.phase
        super()._drive_grab(dt)
        if self.phase in (HOLD, CARRY):
            if phase_before == GRAB or not np.isfinite(self.t_hold):
                self.t_hold = self.time
            f = min(1.0, (self.time - self.t_hold) / self.gp.lift_s)
            self._set_root_dir(self.gp.arm, np.radians(self.gp.lift_pitch_deg) * 0.5 * (1 - np.cos(np.pi * f)))

    def jet(self, dv: float | None = None) -> None:
        super().jet(self.gp.carry_jet_dv if dv is None else dv)

    # -- tick ---------------------------------------------------------------------
    def pre_tick(self) -> None:
        self._drive_grab(self.chain.dt_tick)
        Octopus3D.pre_tick(self)

    def post_tick(self) -> None:
        Octopus3D.post_tick(self)
        self.log_prop.append(self.prop.x.copy())
        self.log_prop3.append(self.prop.x3.copy())
        self.log_phase.append(self.phase)
        self.log_ngrab.append(self.n_sealed())
        self.log_grip.append(0.0)


def make_grab_octopus3d(row: Row, seed: int = 3, direction=(1.0, 0.0), world: World | None = None) -> GrabOctopus3D:
    crawl = replace(row.crawl, seed=seed)
    o = GrabOctopus3D(row.geom, row.chain, crawl, row.body, world=world, grab=row.grab)
    o.set_travel(direction)
    return o
