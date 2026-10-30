"""P5z: the P2 crawl in 3-D at game scale. The hub (lottery, PUSH_STEP,
steering) is `crawl_sim.Octopus`'s, inherited unchanged; only the body, the
arms and the logging move to the 3-D bank (`arm_bank3d.World`).

The animal lies on the deck: the crown centre (the body point) and every node
at z = 0, under g_eff, held up by the floor contact, with Coulomb friction mu
on the deck (the 2-D bank's velocity-space friction on a constant normal load
becomes the floor's position-based friction on the actual penetration -- the
same law for a node resting under g_eff, and a real one when a node is pressed
or lifted).

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

import numpy as np

from arm_bank3d import BankView, BodyView, World
from arm_chain import N_NODES, ArmGeometry
from arm_chain3d import ChainParams3D
from crawl_sim import ARM_ANGLES, STANCE, ArmState, CrawlParams, Octopus
from game_scale import BodyParams, Row


class Octopus3D(Octopus):
    """Eight 3-D arms on a body with height. `pre_tick` = the 2-D hub's
    decision + primitives, `post_tick` = logging; a `World` steps the bank
    between them (several animals may share one world)."""

    def __init__(self, geom: ArmGeometry, chain: ChainParams3D, crawl: CrawlParams, body: BodyParams,
                 world: World | None = None, x0=(0.0, 0.0), heading: float = 0.0, z0: float = 0.0,
                 dead: tuple[int, ...] = (), mu: float | None = None):
        self.geom, self.chain, self.crawl, self.bodyp = geom, chain, crawl, body
        self.world = world if world is not None else World(geom, chain)
        bank = self.world.bank
        mu = crawl.mu_k if mu is None else mu
        b = bank.add_body((x0[0], x0[1], z0), heading, mass=body.mass, radius=body.radius, c_drag=body.c_drag,
                          area=body.crawl_area, rot_damp=body.rot_damp, mu=mu)
        self.body = BodyView(bank, b)
        rest = geom.rest_lengths()
        p0 = np.zeros((8, N_NODES, 3))
        root_local = np.zeros((8, 2, 3))
        c, s = np.cos(heading), np.sin(heading)
        for a, ang in enumerate(ARM_ANGLES):
            u = np.array([np.cos(ang), np.sin(ang), 0.0])
            root_local[a, 0] = body.radius * u
            root_local[a, 1] = (body.radius + rest[0]) * u
            r = body.radius + np.concatenate([[0.0], np.cumsum(rest)])
            uw = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], 0.0])
            p0[a] = np.array([x0[0], x0[1], z0]) + r[:, None] * uw[None, :]
        sl = bank.add_arms(p0, body=b, root_local=root_local, mu=mu, grip=crawl.grip_limit)
        self.bank = BankView(bank, sl)
        for a in dead:
            self.bank.alive[a] = False
        self.rng = np.random.default_rng(crawl.seed)
        self.arms = [ArmState() for _ in range(8)]
        self.travel = np.array([1.0, 0.0])
        self.time = 0.0
        self.t_since_decision = 0.0
        self.n_anchor = max(crawl.anchor_nodes) + 1
        self.log_t, self.log_x, self.log_theta, self.log_stance = [], [], [], []
        self.log_pred, self.log_p, self.log_anchored, self.log_z = [], [], [], []
        self.steps: list[tuple[int, float, float]] = []
        self.world.add(self)

    @property
    def root_local(self) -> np.ndarray:
        return self.bank.bank.root_local[self.bank.sl]

    # -- the tick, split around the bank step ----------------------------------
    def pre_tick(self) -> None:
        dt = self.chain.dt_tick
        self.t_since_decision += dt
        if self.t_since_decision >= self.crawl.decision_s:
            self.t_since_decision = 0.0
            self._decide()
        self._drive_arms(dt)

    def post_tick(self) -> None:
        self.time += self.chain.dt_tick
        stance = np.array([self.arms[a].state == STANCE for a in range(8)])
        pred = np.zeros(2)
        for a in range(8):
            if stance[a]:
                pred += self.arms[a].push_dir
        self.log_t.append(self.time)
        self.log_x.append(self.body.x.copy())
        self.log_z.append(self.body.z)
        self.log_theta.append(self.body.theta)
        self.log_stance.append(stance)
        self.log_pred.append(pred)
        self.log_p.append(self.bank.p.copy())
        self.log_anchored.append(self.bank.anchored.copy())

    def tick(self) -> None:
        assert self.world.animals == [self], "a shared world is stepped by World.run"
        self.world.tick()

    def run(self, seconds: float) -> None:
        self.world.run(seconds)


def make_octopus3d(row: Row, direction=(1.0, 0.0), heading: float = 0.0, dead=(), seed: int = 1,
                   world: World | None = None, crawl: CrawlParams | None = None, x0=(0.0, 0.0)) -> Octopus3D:
    crawl = crawl or row.crawl
    from dataclasses import replace
    crawl = replace(crawl, seed=seed)
    o = Octopus3D(row.geom, row.chain, crawl, row.body, world=world, x0=x0, heading=heading, dead=dead)
    o.set_travel(direction)
    return o
