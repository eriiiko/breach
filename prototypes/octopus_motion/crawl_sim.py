"""P2: the crawl. Eight arms on a floor (seen from above), a rigid body, the
PUSH_STEP primitive (Levy, Flash & Hochner 2015) and the hub's seeded
recruitment lottery (strand (d) recommendation; Levy & Hochner 2017: no
clock, no stepping order; Katija 2025: ~3 arms in contact, release when the
proximal arm's strain saturates).

PUSH_STEP on arm i (Levy 2015 Fig. 4): a patch of MID-ARM suckers anchors at
their current place; the segments between the mouth and the anchor ELONGATE
(rest length x e_max over push_s); the body is pushed away from the anchor;
the anchor releases when the elongation saturates (strain trigger) or the
grip limit pops; the proximal arm then relaxes back to rest length over
recover_s while the arm is in refractory. Distal to the anchor nothing is
commanded. Every arm pushes along its own fixed direction (away from its
root), so the body's velocity is the sum of the pushing arms' directions --
that is the gate.

HUB (no oscillator): every decision tick, while fewer than `max_stance` arms
are in stance, recruit one READY arm with probability p_recruit by a seeded
lottery with weights max(0, push_i . d - s_min) x (1 + back_bias x
backness_i) x (1 + neighbour_bonus if a neighbour is in stance). Dead arms
are never READY. Step durations come out random -> flat spectrum.

Author: Claude (octopus lead), 2026-10-29.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm_bank import ArmBank, Body2D
from arm_chain import N_NODES, N_SEG, ArmGeometry, ChainParams

ARM_NAMES = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]   # around the crown, counter-clockwise from front-left
# angle of each arm's root around the crown, from the heading (+x), counter-clockwise
ARM_ANGLES = np.radians([22.5, 67.5, 112.5, 157.5, -157.5, -112.5, -67.5, -22.5])

READY, STANCE, RECOVER, HOLDING = 0, 1, 2, 3   # HOLDING: driven by another primitive (P4), out of the lottery


@dataclass(frozen=True)
class CrawlParams:
    anchor_nodes: tuple[int, ...] = (5, 6, 7)   # the mid-arm sucker patch, s = 0.3-0.4 L
    e_max: float = 1.6                           # proximal elongation at release (Levy: 1.5-1.8)
    push_s: float = 0.5                          # Levy: active push 0.7 s on average ...
    push_jitter: float = 0.5                     # ... + U(0, jitter): the strain trigger depends on load, never a timer
    recover_s: float = 0.8                       # relax + refractory before READY again
    recover_jitter: float = 2.5                  # + U(0, jitter) s, seeded -> no rhythm (Levy: a step 3.6 s, random)
    decision_s: float = 0.1
    p_recruit: float = 0.3                       # per decision tick, when below max_stance
    max_stance: int = 3
    s_min: float = 0.2                           # an arm must point at least this well (cosine) against travel
    back_bias: float = 0.5                       # hind arms used 1.5x more (Levy)
    neighbour_bonus: float = 0.3
    grip_limit: float = 0.5                      # N per anchored node (a sucker patch of 3)
    mu_k: float = 0.1                            # Coulomb on the deck
    stance_activation: float = 0.3               # proximal stiffening while pushing (keeps the pushed segment straight)
    steer_gain: float = 1.0                      # the hub aims at wanted + gain x (wanted - recent travel direction)
    steer_window_s: float = 2.0
    steer_min_dist: float = 0.02                  # m: below this recent travel the hub steers on the wanted direction alone
    seed: int = 1


@dataclass
class ArmState:
    state: int = READY
    t_in_state: float = 0.0
    recover_len: float = 1.2
    push_len: float = 0.7
    push_dir: np.ndarray = field(default_factory=lambda: np.zeros(2))   # world, at recruit


class Octopus:
    """Bank of 8 arms + body + hub. `set_travel(d)` is the only command."""

    def __init__(self, geom: ArmGeometry, chain: ChainParams, crawl: CrawlParams,
                 x0=(0.0, 0.0), heading: float = 0.0, dead: tuple[int, ...] = ()):
        self.geom, self.chain, self.crawl = geom, chain, crawl
        self.body = Body2D(np.array(x0, dtype=float), heading)
        self.rng = np.random.default_rng(crawl.seed)
        rest = geom.rest_lengths()
        # arms straight out radially from the crown, lying on the floor
        p0 = np.zeros((8, N_NODES, 2))
        root_local = np.zeros((8, 2, 2))
        for a, ang in enumerate(ARM_ANGLES):
            u = np.array([np.cos(ang), np.sin(ang)])
            root_local[a, 0] = self.body.radius * u
            root_local[a, 1] = (self.body.radius + rest[0]) * u
            s = self.body.radius + np.concatenate([[0.0], np.cumsum(rest)])
            p0[a] = self.body.x + (s[:, None] * u[None, :]) @ self.body.rot().T
        self.root_local = root_local
        self.bank = ArmBank(geom, chain, p0)
        self.bank.mu_k = crawl.mu_k
        self.bank.g_normal = 9.81 * (1.0 - geom.rho_water / geom.rho_arm)   # in water: net weight only
        self.bank.grip_limit = crawl.grip_limit
        self.arms = [ArmState() for _ in range(8)]
        for a in dead:
            self.bank.alive[a] = False
        self.travel = np.array([1.0, 0.0])
        self.time = 0.0
        self.t_since_decision = 0.0
        self.n_anchor = max(crawl.anchor_nodes) + 1    # segments 0..n_anchor-1 elongate
        # logging
        self.log_t: list[float] = []
        self.log_x: list[np.ndarray] = []
        self.log_theta: list[float] = []
        self.log_stance: list[np.ndarray] = []
        self.log_pred: list[np.ndarray] = []
        self.log_p: list[np.ndarray] = []
        self.log_anchored: list[np.ndarray] = []
        self.steps: list[tuple[int, float, float]] = []   # (arm, t_start, t_release)

    # -- command -------------------------------------------------------------
    def set_travel(self, direction) -> None:
        d = np.asarray(direction, dtype=float)
        self.travel = d / max(np.linalg.norm(d), 1e-12)

    # -- geometry helpers ----------------------------------------------------
    def push_dirs(self) -> np.ndarray:
        """(8, 2) world push direction of each arm = away from its root."""
        R = self.body.rot()
        u = np.stack([np.cos(ARM_ANGLES), np.sin(ARM_ANGLES)], axis=1) @ R.T
        return -u

    def elongation(self, a: int) -> float:
        return float(self.bank.rest_scale[a, : self.n_anchor].mean())

    # -- the hub -------------------------------------------------------------
    def _decide(self) -> None:
        c = self.crawl
        in_stance = [a for a in range(8) if self.arms[a].state == STANCE]
        if len(in_stance) >= c.max_stance:
            return
        if self.rng.random() >= c.p_recruit:
            return
        dirs = self.push_dirs()
        target = self._steer_target()
        weights = np.zeros(8)
        for a in range(8):
            st = self.arms[a]
            if st.state != READY or not self.bank.alive[a]:
                continue
            align = float(dirs[a] @ target)
            w = max(0.0, align - c.s_min)
            if w <= 0.0:
                continue
            backness = 0.5 * (1.0 - np.cos(ARM_ANGLES[a]))       # 0 front .. 1 back
            w *= 1.0 + c.back_bias * backness
            if self.arms[(a - 1) % 8].state == STANCE or self.arms[(a + 1) % 8].state == STANCE:
                w *= 1.0 + c.neighbour_bonus
            weights[a] = w
        if weights.sum() <= 0.0:
            return
        a = int(self.rng.choice(8, p=weights / weights.sum()))
        self._recruit(a, dirs[a])

    def _steer_target(self) -> np.ndarray:
        """Steering = which arms push (Levy & Hochner 2017): the lottery aims at the
        wanted direction corrected by where the body actually went lately."""
        c = self.crawl
        n = int(round(c.steer_window_s / self.chain.dt_tick))
        if c.steer_gain <= 0.0 or len(self.log_x) <= n:
            return self.travel
        recent = self.log_x[-1] - self.log_x[-1 - n]
        dist = np.linalg.norm(recent)
        if dist < c.steer_min_dist:
            return self.travel
        err = self.travel - recent / dist
        t = self.travel + c.steer_gain * err
        return t / max(np.linalg.norm(t), 1e-12)

    def _recruit(self, a: int, push_dir: np.ndarray) -> None:
        st = self.arms[a]
        st.state, st.t_in_state = STANCE, 0.0
        st.push_dir = push_dir.copy()
        st.push_len = self.crawl.push_s + self.crawl.push_jitter * self.rng.random()
        self.bank.anchor(a, list(self.crawl.anchor_nodes))
        self.bank.activation[a, : self.n_anchor] = self.crawl.stance_activation
        self.steps.append((a, self.time, np.nan))

    def _release(self, a: int) -> None:
        st = self.arms[a]
        st.state, st.t_in_state = RECOVER, 0.0
        st.recover_len = self.crawl.recover_s + self.crawl.recover_jitter * self.rng.random()
        self.bank.release(a)
        self.bank.activation[a, :] = 0.0
        for i in range(len(self.steps) - 1, -1, -1):
            if self.steps[i][0] == a and np.isnan(self.steps[i][2]):
                self.steps[i] = (a, self.steps[i][1], self.time)
                break

    # -- primitives, per tick -------------------------------------------------
    def _drive_arms(self, dt: float) -> None:
        c = self.crawl
        for a in range(8):
            st = self.arms[a]
            st.t_in_state += dt
            if st.state == STANCE:
                frac = min(1.0, st.t_in_state / st.push_len)
                self.bank.rest_scale[a, : self.n_anchor] = 1.0 + (c.e_max - 1.0) * frac
                anchored = self.bank.anchored[a, list(c.anchor_nodes)]
                if frac >= 1.0 or not anchored.any():
                    self._release(a)
            elif st.state == RECOVER:
                frac = min(1.0, st.t_in_state / (0.5 * st.recover_len))
                e = self.elongation(a)
                self.bank.rest_scale[a, : self.n_anchor] = c.e_max + (1.0 - c.e_max) * frac if e > 1.0 else 1.0
                if st.t_in_state >= st.recover_len:
                    st.state, st.t_in_state = READY, 0.0
                    self.bank.rest_scale[a, : self.n_anchor] = 1.0

    def tick(self) -> None:
        dt = self.chain.dt_tick
        self.t_since_decision += dt
        if self.t_since_decision >= self.crawl.decision_s:
            self.t_since_decision = 0.0
            self._decide()
        self._drive_arms(dt)
        self.bank.tick(self.body, self.root_local)
        self.time += dt
        # log
        stance = np.array([self.arms[a].state == STANCE for a in range(8)])
        pred = np.zeros(2)
        for a in range(8):
            if stance[a]:
                pred += self.arms[a].push_dir
        self.log_t.append(self.time)
        self.log_x.append(self.body.x.copy())
        self.log_theta.append(self.body.theta)
        self.log_stance.append(stance)
        self.log_pred.append(pred)
        self.log_p.append(self.bank.p.copy())
        self.log_anchored.append(self.bank.anchored.copy())

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.chain.dt_tick))):
            self.tick()


def make_octopus(direction=(1.0, 0.0), heading: float = 0.0, dead=(), seed: int = 1,
                 chain: ChainParams | None = None, crawl: CrawlParams | None = None) -> Octopus:
    chain = chain or ChainParams(n_sub=16, n_pass=2)
    crawl = crawl or CrawlParams()
    crawl = CrawlParams(**{**crawl.__dict__, "seed": seed})
    o = Octopus(ArmGeometry(), chain, crawl, heading=heading, dead=dead)
    o.set_travel(direction)
    return o
