"""P4: reach -> grab -> hold -> pull in -> carry a marine-shaped prop, on the
crawl animal (top-down). The prop is a rigid disc on the deck; the grab is a
set of rigid attachments between mid-arm nodes and the prop (the sucker
patch), with the bank's grip limit; the hold is a wrap (curvature targets
distal of the grab, toward the prop); the pull-in shortens the proximal
segments so the prop rides under the crown; the carry is the P2 crawl with
the holding arm excluded from the lottery, then jets in the plane.

REACH here is the crawl animal's version: the root segment is aimed at the
target in the body frame (the brain's aim, Yekutieli II), the stiffening
wave runs base -> tip (P1) and the arm ELONGATES toward the target
(stretch); it stops on contact -- any node of the mid-arm patch within the
prop's surface + a sucker's reach. "Lift" is out of this plane: in the game
it is the z of the chain (brief section 6, open question 2) -- here the
pull-in stands for it.

Author: Claude (octopus lead), 2026-10-29.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_bank import Body2D
from arm_chain import N_NODES, N_SEG
from crawl_sim import HOLDING, CrawlParams, Octopus


@dataclass(frozen=True)
class GrabParams:
    arm: int = 0                       # L1, the front-left arm
    prop_mass: float = 0.1             # kg: a marine is ~0.3 of the octopus (90 / 300 kg at game scale)
    prop_radius: float = 0.04          # m, seen from above
    # world, from the body centre (x along the heading): 0.24 m out at 50 deg -- clear of L1 at rest (22.5 deg,
    # 0.11 m lateral gap) so the arm must AIM and STRETCH; contact lands mid-arm (s ~ 0.55 L at 1.4x)
    prop_at: tuple[float, float] = (0.24 * np.cos(np.radians(50)), 0.24 * np.sin(np.radians(50)))
    reach_wave_s: float = 0.8
    reach_stretch: float = 1.4
    reach_activation: float = 0.6
    grab_nodes: tuple[int, int] = (3, 11)          # the patch that may seal: s = 0.18 .. 0.65 L
    sucker_reach: float = 0.012                    # m beyond the prop surface that counts as contact
    wrap_activation: float = 0.6
    hold_activation: float = 1.0                   # the proximal arm stiffens to carry (Yekutieli II: force = stiffness)
    pull_in_scale: float = 0.7
    pull_in_s: float = 1.0
    grip_limit: float = 1.5                        # N per sealed node: a held prop needs more than a crawl anchor
    hold_s: float = 1.5                            # from grab to carry
    aim_max_deg: float = 60.0                      # how far the root can swing toward the target


IDLE, AIM_REACH, GRAB, HOLD, CARRY = "idle", "reach", "grab", "hold", "carry"


class GrabOctopus(Octopus):
    def __init__(self, *args, grab: GrabParams | None = None, **kw):
        super().__init__(*args, **kw)
        self.gp = grab or GrabParams()
        g = self.gp
        self.prop = Body2D(self.body.x + np.array(g.prop_at), 0.0, mass=g.prop_mass, radius=g.prop_radius,
                           c_drag=1.0, frontal_area=2 * g.prop_radius * 0.05)
        self.prop.w = 1.0 / g.prop_mass
        self.prop.inertia = 0.5 * g.prop_mass * g.prop_radius ** 2
        self.prop.w_rot = 1.0 / self.prop.inertia
        self.phase = IDLE
        self.t_phase = 0.0
        self.t_grab = np.nan
        self.grab_s = []                            # arc lengths of the sealed nodes at the grab
        self.root_dir0 = self.root_local[g.arm, 1] - self.root_local[g.arm, 0]
        self.log_prop: list[np.ndarray] = []
        self.log_phase: list[str] = []
        self.log_ngrab: list[int] = []
        self.log_grip: list[float] = []

    # -- commands --------------------------------------------------------------
    def start_reach(self) -> None:
        self.phase, self.t_phase = AIM_REACH, 0.0
        a = self.gp.arm
        self.arms[a].state = HOLDING     # out of the lottery and out of the push-step from here on
        self.bank.release(a)

    # -- helpers -------------------------------------------------------------
    def _aim_root(self, a: int, target_world: np.ndarray) -> None:
        """Swing the root segment toward the target inside the body frame, up to aim_max."""
        R = self.body.rot()
        root_w = self.body.x + self.root_local[a, 0] @ R.T
        d = (target_world - root_w) @ R          # into the body frame
        d /= max(np.linalg.norm(d), 1e-12)
        u0 = self.root_dir0 / np.linalg.norm(self.root_dir0)
        ang = np.arctan2(u0[0] * d[1] - u0[1] * d[0], u0 @ d)
        ang = float(np.clip(ang, -np.radians(self.gp.aim_max_deg), np.radians(self.gp.aim_max_deg)))
        c, s = np.cos(ang), np.sin(ang)
        u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1]])
        self.root_local[a, 1] = self.root_local[a, 0] + np.linalg.norm(self.root_dir0) * u

    def _contact_nodes(self, a: int) -> np.ndarray:
        k0, k1 = self.gp.grab_nodes
        d = np.linalg.norm(self.bank.p[a, k0:k1 + 1] - self.prop.x, axis=1)
        return np.arange(k0, k1 + 1)[d <= self.prop.radius + self.gp.sucker_reach]

    def _seal(self, a: int, nodes: np.ndarray) -> None:
        R = self.prop.rot()
        for k in nodes:
            if any(g[0] == a and g[1] == k for g in self.bank.grabs):
                continue
            local = (self.bank.p[a, k] - self.prop.x) @ R
            self.bank.grabs.append((a, int(k), self.prop, local))

    def n_sealed(self) -> int:
        return len(self.bank.grabs)

    # -- the state machine, per tick ----------------------------------------------
    def _drive_grab(self, dt: float) -> None:
        g, a = self.gp, self.gp.arm
        self.t_phase += dt
        if self.phase == AIM_REACH:
            self._aim_root(a, self.prop.x)
            # the P1 wave: activation front base -> tip over reach_wave_s, then held
            L = self.geom.length_m
            s_front = L * self.t_phase / g.reach_wave_s
            w = L / N_SEG
            self.bank.activation[a] = g.reach_activation * 0.5 * (1 + np.tanh((s_front - self.bank.s_node[1:-1]) / w))
            self.bank.theta_target[a] = 0.0
            f = min(1.0, self.t_phase / g.reach_wave_s)
            self.bank.rest_scale[a, :] = 1.0 + (g.reach_stretch - 1.0) * f
            hit = self._contact_nodes(a)
            if len(hit) >= 1:
                self.phase, self.t_phase = GRAB, 0.0
                self.t_grab = self.time
                self.bank.grip_limit = g.grip_limit
                self._seal(a, hit)
                self.grab_s = [float(self.bank.s_node[k]) for k in hit]
        elif self.phase == GRAB:
            # the wrap: distal of the first sealed node, curl toward the prop; seal whatever touches
            k_first = min(k for (aa, k, _, _) in self.bank.grabs if aa == a)
            side = self._wrap_side(a, k_first)
            turn = self.geom.rest_lengths()[0] / self.prop.radius      # one segment's turn on the prop's rim
            self.bank.theta_target[a, max(0, k_first - 1):] = side * turn
            self.bank.activation[a, :] = g.wrap_activation
            self.bank.rest_scale[a, k_first:] = 1.0
            hit = self._contact_nodes(a)
            self._seal(a, hit)
            if self.t_phase >= 0.6:
                self.phase, self.t_phase = HOLD, 0.0
        elif self.phase == HOLD:
            k_first = min(k for (aa, k, _, _) in self.bank.grabs if aa == a)
            f = min(1.0, self.t_phase / g.pull_in_s)
            self.bank.rest_scale[a, :k_first] = g.reach_stretch + (g.pull_in_scale - g.reach_stretch) * f
            self.bank.activation[a, :k_first] = g.hold_activation
            if self.t_phase >= g.hold_s:
                self.phase, self.t_phase = CARRY, 0.0
        elif self.phase == CARRY:
            pass   # the hub crawls; the holding arm stays out of the lottery (STANCE that never pushes)

    def _wrap_side(self, a: int, k: int) -> float:
        p = self.bank.p[a]
        d = p[min(k + 1, N_NODES - 1)] - p[max(k - 1, 0)]
        to_prop = self.prop.x - p[k]
        return 1.0 if d[0] * to_prop[1] - d[1] * to_prop[0] > 0 else -1.0

    def jet(self, dv: float = 0.5) -> None:
        h = np.array([np.cos(self.body.theta), np.sin(self.body.theta)])
        self.body.v += dv * h

    def tick(self) -> None:
        self._drive_grab(self.chain.dt_tick)
        super().tick()
        self.log_prop.append(self.prop.x.copy())
        self.log_phase.append(self.phase)
        self.log_ngrab.append(self.n_sealed())
        self.log_grip.append(0.0)


def make_grab_octopus(seed: int = 3, direction=(1.0, 0.0), grab: GrabParams | None = None) -> GrabOctopus:
    from arm_chain import ArmGeometry, ChainParams
    chain = ChainParams(n_sub=16, n_pass=2)
    crawl = CrawlParams(seed=seed)
    o = GrabOctopus(ArmGeometry(), chain, crawl, grab=grab)
    o.set_travel(direction)
    return o
