"""P3: swimming. The umbrella (medusoid) stroke and the jet, in water, seen
from the side: the body axis lies in the plane, the eight arms project onto
an UP group and a DOWN group of four (one bank arm each, mass and drag x 4,
the web as a drag factor on the arms). The animal swims mantle-first: +heading
is where it goes, the arms trail behind.

UMBRELLA (Huffard 2006; Sfakiotakis / FORTH 2015): the arms (with the web)
open SLOWLY to ~80 deg from the trailing axis and close FAST to ~20 deg,
open:close = 2.5:1. The root segment's direction is a commanded pose on the
rigid body (root_local[:, 1] moves in the body frame); the rest of the arm is
a stiff paddle (activation on every bend, target 0) whose tip lags. The body
receives ONLY the arms' attachment reactions and its own drag -- so if the
water drag were linear in speed the out-and-back sweep would net nothing
(the soft-arm digest §4.0): that is the control.

JET (Wells 1987 / Trueman & Packard 1968, timing still unread): a force on
the body along +heading for jet_s seconds, arms bundled (root angle -> ~5
deg, stiff). Thrust from the impulse I = M dv_jet.

Author: Claude (octopus lead), 2026-10-29.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_bank import ArmBank, Body2D
from arm_chain import N_NODES, ArmGeometry, ChainParams


@dataclass(frozen=True)
class SwimParams:
    open_deg: float = 80.0          # arm root angle from the trailing axis, umbrella open
    closed_deg: float = 20.0        # ... closed
    bundled_deg: float = 5.0        # during a jet
    open_s: float = 1.0             # slow opening
    close_s: float = 0.4            # fast closing (2.5:1)
    activation: float = 0.8         # the paddle's stiffness
    web_factor: float = 3.0         # the web's broadside area per arm, as a multiple of the bare arm's
    arms_per_group: int = 4
    jet_dv: float = 0.8             # m/s the mantle impulse gives the body (Wells: 45-100 cm/s)
    jet_s: float = 0.3
    body_c_drag: float = 0.5        # mantle-first, streamlined
    body_area: float = np.pi * 0.05 ** 2


class Swimmer:
    """Two bank arms (up / down group) on a body in open water."""

    def __init__(self, geom: ArmGeometry, chain: ChainParams, swim: SwimParams,
                 linear_drag_v_ref: float | None = None):
        self.geom, self.chain, self.swim = geom, chain, swim
        self.body = Body2D(np.zeros(2), 0.0, c_drag=swim.body_c_drag, frontal_area=swim.body_area)
        rest = geom.rest_lengths()
        self.crown_local = np.array([-self.body.radius, 0.0])        # the arms root at the back of the head
        self.side = np.array([1.0, -1.0])                             # up group, down group
        p0 = np.zeros((2, N_NODES, 2))
        self.root_local = np.zeros((2, 2, 2))
        phi0 = np.radians(swim.open_deg)
        for a in range(2):
            d = self._arm_dir(phi0, a)
            self.root_local[a, 0] = self.crown_local
            self.root_local[a, 1] = self.crown_local + rest[0] * d
            s = np.concatenate([[0.0], np.cumsum(rest)])
            p0[a] = self.body.x + self.crown_local + s[:, None] * d[None, :]
        self.bank = ArmBank(geom, chain, p0)
        n = swim.arms_per_group
        self.bank.m *= n
        self.bank.w = 1.0 / self.bank.m
        self.bank.K_per *= n * swim.web_factor
        self.bank.K_tan *= n
        self.bank.linear_drag_v_ref = linear_drag_v_ref
        self.bank.activation[:] = swim.activation
        self.phi = phi0
        self.time = 0.0
        self.mode = "idle"
        self.t_mode = 0.0
        self.jet_force = 0.0
        self.log_t: list[float] = []
        self.log_x: list[np.ndarray] = []
        self.log_phi: list[float] = []
        self.log_p: list[np.ndarray] = []
        self.log_phase: list[str] = []

    def _arm_dir(self, phi: float, a: int) -> np.ndarray:
        """Root direction in the body frame: phi from the trailing axis (-x), toward +-y."""
        return np.array([-np.cos(phi), self.side[a] * np.sin(phi)])

    def _set_root(self, phi: float) -> None:
        rest0 = self.geom.rest_lengths()[0]
        for a in range(2):
            self.root_local[a, 1] = self.crown_local + rest0 * self._arm_dir(phi, a)
        self.phi = phi

    # -- commands ------------------------------------------------------------
    def start_umbrella(self) -> None:
        self.mode, self.t_mode = "open", 0.0

    def start_jet(self) -> None:
        self.mode, self.t_mode = "jet", 0.0

    def stop(self) -> None:
        self.mode, self.t_mode = "idle", 0.0

    # -- one tick --------------------------------------------------------------
    def tick(self) -> None:
        self._command()
        self.bank.tick(self.body, self.root_local)
        self._log()

    def _command(self) -> None:
        """The umbrella / jet state machine: this tick's root angle and jet impulse (P5z reuses it in 3-D)."""
        sw, dt = self.swim, self.chain.dt_tick
        self.t_mode += dt
        o, c = np.radians(sw.open_deg), np.radians(sw.closed_deg)
        if self.mode == "open":
            f = min(1.0, self.t_mode / sw.open_s)
            self._set_root(c + (o - c) * 0.5 * (1 - np.cos(np.pi * f)))
            if f >= 1.0:
                self.mode, self.t_mode = "close", 0.0
        elif self.mode == "close":
            f = min(1.0, self.t_mode / sw.close_s)
            self._set_root(o + (c - o) * 0.5 * (1 - np.cos(np.pi * f)))
            if f >= 1.0:
                self.mode, self.t_mode = "open", 0.0
        elif self.mode == "jet":
            f = min(1.0, self.t_mode / sw.jet_s)
            self._set_root(self.phi + (np.radians(sw.bundled_deg) - self.phi) * min(1.0, 4.0 * dt / sw.jet_s))
            self.jet_force = self.body.mass * sw.jet_dv / sw.jet_s if f < 1.0 else 0.0
            if f >= 1.0:
                self.mode, self.t_mode = "coast", 0.0
        if self.mode != "jet":
            self.jet_force = 0.0
        # the jet impulse enters the body's velocity before the arms' substeps
        if self.jet_force:
            h = np.array([np.cos(self.body.theta), np.sin(self.body.theta)])
            self.body.v += dt * self.jet_force * self.body.w * h

    def _log(self) -> None:
        self.time += self.chain.dt_tick
        self.log_t.append(self.time)
        self.log_x.append(self.body.x.copy())
        self.log_phi.append(self.phi)
        self.log_p.append(self.bank.p.copy())
        self.log_phase.append(self.mode)

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.chain.dt_tick))):
            self.tick()


def make_swimmer(linear: bool = False, chain: ChainParams | None = None, swim: SwimParams | None = None,
                 v_ref: float = 0.3) -> Swimmer:
    chain = chain or ChainParams(n_sub=16, n_pass=2)
    swim = swim or SwimParams()
    return Swimmer(ArmGeometry(), chain, swim, linear_drag_v_ref=(v_ref if linear else None))
