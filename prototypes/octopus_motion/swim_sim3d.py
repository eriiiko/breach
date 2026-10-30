"""P5z: the P3 swim in the real 3-D chain at game scale. The umbrella / jet
state machine is `swim_sim.Swimmer._command`, inherited unchanged; what moves to
3-D is the animal: eight real arms (not the 2-D side view's two grouped arms),
each rooted on a small ring at the back of the head and commanded by a ROOT
ANGLE phi from the trailing axis, toward the arm's own radial direction psi_a
around the body axis:

    root direction (body frame) = (-cos phi, sin phi cos psi_a, sin phi sin psi_a)

so the umbrella opens and closes as a cone about the swimming axis. The web is
the 2-D model's: a x web_factor broadside drag on every arm. The body swims
mantle first along its heading at mid water (z0 well above the deck), under
g_eff -- it sinks slowly while it swims; the gates read the horizontal travel
and report the sink.

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

import numpy as np

from arm_bank3d import DRAG_LINEAR, BankView, BodyView, World
from arm_chain import N_NODES
from crawl_sim import ARM_ANGLES
from game_scale import Row
from swim_sim import SwimParams, Swimmer


class Swimmer3D(Swimmer):
    RING = 0.3          # root ring radius, as a fraction of the crown radius

    def __init__(self, row: Row, world: World | None = None, z0: float = 8.0, linear_drag_v_ref: float | None = None,
                 x0=(0.0, 0.0)):
        geom, chain, swim, bp = row.geom, row.chain, row.swim, row.body
        self.geom, self.chain, self.swim = geom, chain, swim
        self.world = world if world is not None else World(geom, chain)
        bank = self.world.bank
        b = bank.add_body((x0[0], x0[1], z0), 0.0, mass=bp.mass, radius=bp.radius, c_drag=swim.body_c_drag,
                          area=swim.body_area, rot_damp=bp.rot_damp, mu=0.0)
        self.body = BodyView(bank, b)
        if linear_drag_v_ref is not None:
            bank.bdrag_mode[b] = DRAG_LINEAR
            bank.bvref[b] = linear_drag_v_ref
        self.psi = np.array(ARM_ANGLES)                 # each arm's direction around the swimming axis
        self.crown_local = np.array([-bp.radius, 0.0, 0.0])
        rest = geom.rest_lengths()
        phi0 = np.radians(swim.open_deg)
        p0 = np.zeros((8, N_NODES, 3))
        root_local = np.zeros((8, 2, 3))
        s = np.concatenate([[0.0], np.cumsum(rest)])
        for a in range(8):
            ring = self.crown_local + self.RING * bp.radius * np.array([0.0, np.cos(self.psi[a]), np.sin(self.psi[a])])
            d = self._arm_dir(phi0, a)
            root_local[a, 0] = ring
            root_local[a, 1] = ring + rest[0] * d
            p0[a] = np.array([x0[0], x0[1], z0]) + ring + s[:, None] * d[None, :]
        per_arm = dict(mu=0.0)
        if linear_drag_v_ref is not None:
            per_arm.update(drag_mode=DRAG_LINEAR, drag_vref=linear_drag_v_ref)
        sl = bank.add_arms(p0, body=b, root_local=root_local, **per_arm)
        self.bank = BankView(bank, sl)
        self.bank.K_per[:] *= swim.web_factor
        self.bank.activation[:] = swim.activation
        self.phi = phi0
        self.time = 0.0
        self.mode = "idle"
        self.t_mode = 0.0
        self.jet_force = 0.0
        self.log_t, self.log_x, self.log_phi, self.log_p, self.log_phase, self.log_z = [], [], [], [], [], []
        self.world.add(self)

    @property
    def root_local(self) -> np.ndarray:
        return self.bank.bank.root_local[self.bank.sl]

    def _arm_dir(self, phi: float, a: int) -> np.ndarray:
        return np.array([-np.cos(phi), np.sin(phi) * np.cos(self.psi[a]), np.sin(phi) * np.sin(self.psi[a])])

    def _set_root(self, phi: float) -> None:
        rest0 = self.geom.rest_lengths()[0]
        rl = self.root_local
        for a in range(8):
            rl[a, 1] = rl[a, 0] + rest0 * self._arm_dir(phi, a)
        self.phi = phi

    def pre_tick(self) -> None:
        self._command()

    def post_tick(self) -> None:
        self._log()
        self.log_z.append(self.body.z)

    def tick(self) -> None:
        assert self.world.animals == [self]
        self.world.tick()

    def run(self, seconds: float) -> None:
        self.world.run(seconds)


def make_swimmer3d(row: Row, linear: bool = False, v_ref: float | None = None, world: World | None = None,
                   x0=(0.0, 0.0)) -> Swimmer3D:
    v_ref = 0.3 * row.S if v_ref is None else v_ref     # the 2-D control's 0.3 m/s, a speed: x L
    return Swimmer3D(row, world=world, linear_drag_v_ref=(v_ref if linear else None), x0=x0)
