"""P5: idle that reads as alive. The animal sits on the deck; the hub, with no
travel goal, lets one or two arms at a time EXPLORE: a decision hold
(stillness), then a reach (the P1 wave: aim in the body frame + the
stiffening front + a little stretch) toward a random nearby point, a touch
(hold at the end), then LIMP (activation off, the arm relaxes under drag
and passive stiffness), then another hold. Front pairs explore more
(Mather; the motor-biology digest: front pairs probe, back pairs support).
Seeded; no clock. Breathing (mantle_squeeze) and tip curls are render
controls (CONTROLS.md), not chain state, so they are not here.

Scored only by the craft checklist (games/film digest, synthesis section 2)
rows that can be measured: 1 few actors, 3 willed motion from the base,
7 holds and rests, 8 de-synchrony, 13 stable over time.

Author: Claude (octopus lead), 2026-10-29.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_chain import N_SEG
from crawl_sim import ARM_ANGLES, HOLDING, READY, CrawlParams, Octopus


@dataclass(frozen=True)
class IdleParams:
    max_active: int = 2
    p_start: float = 0.15             # per decision tick, when fewer than max_active explore
    hold_s: float = 0.4               # the decide beat before a reach
    hold_jitter: float = 1.2
    wave_s: float = 0.7
    wave_jitter: float = 0.6
    stretch: float = 1.15
    stretch_jitter: float = 0.2
    activation: float = 0.5
    touch_s: float = 0.3
    touch_jitter: float = 0.8
    limp_s: float = 1.0
    limp_jitter: float = 1.5
    aim_deg: float = 45.0             # random aim within +-aim_deg of the arm's own direction
    front_bias: float = 1.0           # weight 1 + front_bias x frontness


STILL, DECIDE, REACH, TOUCH, LIMP = "still", "decide", "reach", "touch", "limp"


class IdleOctopus(Octopus):
    def __init__(self, *args, idle: IdleParams | None = None, **kw):
        super().__init__(*args, **kw)
        self.ip = idle or IdleParams()
        self.ex = [STILL] * 8
        self.ex_t = np.zeros(8)
        self.ex_len = np.zeros(8)
        self.ex_wave = np.full(8, self.ip.wave_s)
        self.ex_stretch = np.ones(8)
        self.root_dir0 = (self.root_local[:, 1] - self.root_local[:, 0]).copy()
        self.gestures: list[tuple[int, float, float]] = []    # (arm, t_reach_start, t_limp_end)
        self.log_ex: list[list[str]] = []
        # a seated octopus is stuck down: every still arm's mid-arm patch is sealed to the deck
        # (~1 in 5 suckers, motor-biology digest), so a reaching arm's reaction never moves the body
        for a in range(8):
            self.bank.anchor(a, list(self.crawl.anchor_nodes))

    def _decide(self) -> None:
        ip = self.ip
        active = [a for a in range(8) if self.ex[a] != STILL]
        if len(active) >= ip.max_active or self.rng.random() >= ip.p_start:
            return
        w = np.zeros(8)
        for a in range(8):
            if self.ex[a] == STILL and self.bank.alive[a]:
                frontness = 0.5 * (1.0 + np.cos(ARM_ANGLES[a]))
                w[a] = 1.0 + ip.front_bias * frontness
        if w.sum() <= 0:
            return
        a = int(self.rng.choice(8, p=w / w.sum()))
        self.bank.release(a)
        self.ex[a], self.ex_t[a] = DECIDE, 0.0
        self.ex_len[a] = ip.hold_s + ip.hold_jitter * self.rng.random()
        self.arms[a].state = HOLDING

    def _aim(self, a: int, ang: float) -> None:
        u0 = self.root_dir0[a] / np.linalg.norm(self.root_dir0[a])
        c, s = np.cos(ang), np.sin(ang)
        u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1]])
        self.root_local[a, 1] = self.root_local[a, 0] + np.linalg.norm(self.root_dir0[a]) * u

    def _drive_arms(self, dt: float) -> None:
        ip = self.ip
        L = self.geom.length_m
        for a in range(8):
            st = self.ex[a]
            if st == STILL:
                continue
            self.ex_t[a] += dt
            t = self.ex_t[a]
            if st == DECIDE:
                if t >= self.ex_len[a]:
                    self.ex[a], self.ex_t[a] = REACH, 0.0
                    self.ex_wave[a] = ip.wave_s + ip.wave_jitter * self.rng.random()
                    self.ex_stretch[a] = ip.stretch + ip.stretch_jitter * self.rng.random()
                    self._aim(a, np.radians(ip.aim_deg) * (2 * self.rng.random() - 1))
                    self.gestures.append((a, self.time, np.nan))
            elif st == REACH:
                s_front = L * t / self.ex_wave[a]
                wdt = L / N_SEG
                self.bank.activation[a] = ip.activation * 0.5 * (1 + np.tanh((s_front - self.bank.s_node[1:-1]) / wdt))
                self.bank.theta_target[a] = 0.0
                f = min(1.0, t / self.ex_wave[a])
                self.bank.rest_scale[a, :] = 1.0 + (self.ex_stretch[a] - 1.0) * f
                if f >= 1.0:
                    self.ex[a], self.ex_t[a] = TOUCH, 0.0
                    self.ex_len[a] = ip.touch_s + ip.touch_jitter * self.rng.random()
            elif st == TOUCH:
                if t >= self.ex_len[a]:
                    self.ex[a], self.ex_t[a] = LIMP, 0.0
                    self.ex_len[a] = ip.limp_s + ip.limp_jitter * self.rng.random()
                    self.bank.activation[a] = 0.0
            elif st == LIMP:
                f = min(1.0, t / max(0.5 * self.ex_len[a], 1e-9))
                self.bank.rest_scale[a, :] = self.ex_stretch[a] + (1.0 - self.ex_stretch[a]) * f
                if t >= self.ex_len[a]:
                    self.ex[a], self.ex_t[a] = STILL, 0.0
                    self.bank.rest_scale[a, :] = 1.0
                    self.arms[a].state = READY
                    self.bank.anchor(a, list(self.crawl.anchor_nodes))     # settle: stick down where it lies
                    for i in range(len(self.gestures) - 1, -1, -1):
                        if self.gestures[i][0] == a and np.isnan(self.gestures[i][2]):
                            self.gestures[i] = (a, self.gestures[i][1], self.time)
                            break

    def tick(self) -> None:
        super().tick()
        self.log_ex.append(list(self.ex))


def make_idle_octopus(seed: int = 5, idle: IdleParams | None = None) -> IdleOctopus:
    from arm_chain import ArmGeometry, ChainParams
    chain = ChainParams(n_sub=16, n_pass=2)
    crawl = CrawlParams(seed=seed, p_recruit=0.0)     # no travel: the crawl hub never recruits
    return IdleOctopus(ArmGeometry(), chain, crawl, idle=idle)
