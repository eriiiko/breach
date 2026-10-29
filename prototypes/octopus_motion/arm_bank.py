"""A bank of A octopus arms stepped together: the P1 chain (`arm_chain.py`)
batched over arms, plus what P2-P4 need -- sucker anchors with a grip limit,
floor friction, per-segment rest-length scaling (elongation), and a rigid
body the arm roots attach to. Pure numpy, planar (the floor plane seen from
above for crawl; any plane for a single arm).

Same physics as ArmChain, constraint for constraint (`test_bank_matches_chain`
in crawl_gates.py holds them equal for one arm); the loops run over the
constraint index with all A arms as a vector, which is the SIMD-over-arms the
port will use.

Author: Claude (octopus lead), 2026-10-29. Technique: XPBD (Macklin 2016),
Small Steps (Macklin 2019), Yekutieli 2005 I/II; anchor-as-attachment with a
grip limit after the soft-arm digest §4.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_chain import N_NODES, N_SEG, ArmGeometry, ChainParams

N_BEND = N_SEG - 1


@dataclass
class Body2D:
    """A rigid disc in the plane: the mantle + head. Moved ONLY by the arms'
    attachment reactions, drag and (later) the jet."""
    x: np.ndarray                      # (2,) m
    theta: float                       # rad, heading (0 = +x)
    mass: float = 0.3                  # kg (O. vulgaris with 24 cm arms)
    radius: float = 0.05               # m, crown radius where the arms root
    c_drag: float = 1.0
    frontal_area: float = 0.1 * 0.05   # m^2 (height x width seen by the flow)
    rot_damp: float = 2e-4             # N m s, viscous yaw damping

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64).copy()
        self.v = np.zeros(2)
        self.omega = 0.0
        self.inertia = 0.5 * self.mass * self.radius ** 2
        self.w = 1.0 / self.mass
        self.w_rot = 1.0 / self.inertia

    def rot(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def world(self, local: np.ndarray) -> np.ndarray:
        return self.x + local @ self.rot().T


class ArmBank:
    def __init__(self, geom: ArmGeometry, params: ChainParams, p0: np.ndarray,
                 rho_fluid: float | None = None):
        """p0: (A, N, 2) initial node positions."""
        self.geom = geom
        self.prm = params
        A = p0.shape[0]
        self.A = A
        self.p = p0.astype(np.float64).copy()
        self.v = np.zeros_like(self.p)
        m1 = geom.masses()
        self.m = np.broadcast_to(m1, (A, N_NODES)).copy()
        self.w = 1.0 / self.m
        self.rest0 = np.broadcast_to(geom.rest_lengths(), (A, N_SEG)).copy()
        self.rest_scale = np.ones((A, N_SEG))           # the muscle's elongation / shortening
        self.radii = geom.radii()
        self.s_node = np.concatenate([[0.0], np.cumsum(geom.rest_lengths())])
        self.theta_target = np.zeros((A, N_BEND))
        r_ratio = self.radii[1:-1] / geom.r_base_m
        self.k_active = params.k_bend * r_ratio ** 3
        self.k_passive = params.k_passive_frac * params.k_bend * r_ratio ** 3
        self.activation = np.zeros((A, N_BEND))
        rho_f = geom.rho_water if rho_fluid is None else rho_fluid
        l_node = np.zeros(N_NODES)
        l_node[:-1] += 0.5 * geom.rest_lengths()
        l_node[1:] += 0.5 * geom.rest_lengths()
        self.K_per = 0.5 * rho_f * (2.0 * self.radii * l_node) * params.c_per
        self.K_tan = 0.5 * rho_f * (2.0 * np.pi * self.radii * l_node) * params.c_tan
        # anchors: a node pinned to a world point, released above a grip limit
        self.anchored = np.zeros((A, N_NODES), dtype=bool)
        self.anchor_pos = np.zeros((A, N_NODES, 2))
        self.anchor_force = np.zeros((A, N_NODES))     # |lambda| / dt^2 of the last substep, N
        self.grip_limit = np.inf                        # N per anchored node
        self.popped = np.zeros((A, N_NODES), dtype=bool)   # set when a grip limit released a node this tick
        # floor: Coulomb sliding friction on every node, normal load = net weight
        self.mu_k = 0.0
        self.g_normal = 0.0                             # m/s^2 of net weight pressing on the floor
        self.alive = np.ones(A, dtype=bool)             # a dead arm (#118): passive, never recruited
        # rigid attachments beyond the roots: (arm, node, Body2D, local point) -- a grab on a prop
        self.grabs: list[tuple[int, int, Body2D, np.ndarray]] = []
        # the P3 control: LINEAR drag matched to the quadratic law at v_ref (None = the real, quadratic law)
        self.linear_drag_v_ref: float | None = None
        self.time = 0.0

    # -- convenience ---------------------------------------------------------
    @property
    def rest(self) -> np.ndarray:
        return self.rest0 * self.rest_scale

    def anchor(self, arm: int, nodes, grip_limit: float | None = None) -> None:
        self.anchored[arm, nodes] = True
        self.anchor_pos[arm, nodes] = self.p[arm, nodes]
        if grip_limit is not None:
            self.grip_limit = grip_limit

    def release(self, arm: int) -> None:
        self.anchored[arm, :] = False

    # -- one substep ---------------------------------------------------------
    def substep(self, dt: float, body: Body2D | None = None, root_local: np.ndarray | None = None) -> None:
        prm = self.prm
        p, v, w = self.p, self.v, self.w
        # drag on the local tangent, implicit per component
        tang = np.empty_like(p)
        tang[:, 1:-1] = p[:, 2:] - p[:, :-2]
        tang[:, 0] = p[:, 1] - p[:, 0]
        tang[:, -1] = p[:, -1] - p[:, -2]
        tang /= np.maximum(np.linalg.norm(tang, axis=2), 1e-12)[..., None]
        norm = np.stack([-tang[..., 1], tang[..., 0]], axis=2)
        v_t = (v * tang).sum(axis=2)
        v_n = (v * norm).sum(axis=2)
        if self.linear_drag_v_ref is None:
            v_t *= 1.0 / (1.0 + dt * self.K_tan * np.abs(v_t) * w)
            v_n *= 1.0 / (1.0 + dt * self.K_per * np.abs(v_n) * w)
        else:
            # the control: linear AND isotropic (K_per at v_ref on both components), so that an
            # out-and-back sweep nets exactly nothing by itself
            k_lin = 1.0 / (1.0 + dt * self.K_per * self.linear_drag_v_ref * w)
            v_t *= k_lin
            v_n *= k_lin
        v[:] = v_t[..., None] * tang + v_n[..., None] * norm
        # floor friction (Coulomb, implicit: cannot reverse the velocity)
        if self.mu_k > 0.0 and self.g_normal > 0.0:
            speed = np.linalg.norm(v, axis=2)
            dv = dt * self.mu_k * self.g_normal
            v *= np.clip(1.0 - dv / np.maximum(speed, 1e-12), 0.0, 1.0)[..., None]
        v[self.anchored] = 0.0
        p_prev = p.copy()
        p += dt * v
        bodies = ([body] if body is not None else []) + list({id(g[2]): g[2] for g in self.grabs}.values())
        prev = {}
        for b in bodies:
            prev[id(b)] = (b.x.copy(), b.theta)
            self._body_predict(b, dt)
        for _ in range(prm.n_pass):
            self._project_distance(dt)
            self._project_bend(dt, p_prev)
            self._project_anchors(dt)
            if body is not None:
                self._project_body(dt, body, root_local)
            for a, k, b, local in self.grabs:
                self._attach(a, k, b, local)
        v[:] = (p - p_prev) / dt
        v[self.anchored] = 0.0
        for b in bodies:
            x0, th0 = prev[id(b)]
            b.v = (b.x - x0) / dt
            b.omega = (b.theta - th0) / dt
        self.time += dt

    def _body_predict(self, b: Body2D, dt: float) -> None:
        v_body = np.linalg.norm(b.v) if self.linear_drag_v_ref is None else self.linear_drag_v_ref
        b.v *= 1.0 / (1.0 + dt * 0.5 * self.geom.rho_water * b.c_drag * b.frontal_area * v_body * b.w)
        if self.mu_k > 0.0 and self.g_normal > 0.0:
            sp = np.linalg.norm(b.v)
            b.v *= max(0.0, 1.0 - dt * self.mu_k * self.g_normal / max(sp, 1e-12))
        b.omega *= 1.0 / (1.0 + dt * b.rot_damp * b.w_rot)
        b.x += dt * b.v
        b.theta += dt * b.omega

    def _project_distance(self, dt: float) -> None:
        p, w = self.p, self.w
        alpha = (1.0 / self.prm.k_dist) / (dt * dt)
        rest = self.rest
        for i in range(N_SEG):
            d = p[:, i + 1] - p[:, i]
            ln = np.sqrt((d * d).sum(axis=1))
            ln = np.maximum(ln, 1e-12)
            C = ln - rest[:, i]
            wsum = w[:, i] + w[:, i + 1]
            dlam = np.where(wsum > 0, -C / (wsum + alpha), 0.0)
            g = d / ln[:, None]
            p[:, i] -= (w[:, i] * dlam)[:, None] * g
            p[:, i + 1] += (w[:, i + 1] * dlam)[:, None] * g

    def _project_bend(self, dt: float, p_prev: np.ndarray) -> None:
        p, w = self.p, self.w
        gamma = self.prm.bend_damp_s / dt
        for j in range(N_BEND):
            i0, i1, i2 = j, j + 1, j + 2
            d1 = p[:, i1] - p[:, i0]
            d2 = p[:, i2] - p[:, i1]
            l1 = np.maximum((d1 * d1).sum(axis=1), 1e-16)
            l2 = np.maximum((d2 * d2).sum(axis=1), 1e-16)
            theta = np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(axis=1))
            C = theta - self.theta_target[:, j]
            g0 = np.stack([-d1[:, 1], d1[:, 0]], axis=1) / l1[:, None]
            g2 = np.stack([-d2[:, 1], d2[:, 0]], axis=1) / l2[:, None]
            g1 = -g0 - g2
            k = self.k_passive[j] + self.activation[:, j] * self.k_active[j]
            alpha = (1.0 / k) / (dt * dt)
            denom = (1.0 + gamma) * (w[:, i0] * (g0 * g0).sum(1) + w[:, i1] * (g1 * g1).sum(1)
                                     + w[:, i2] * (g2 * g2).sum(1)) + alpha
            damp = gamma * ((g0 * (p[:, i0] - p_prev[:, i0])).sum(1) + (g1 * (p[:, i1] - p_prev[:, i1])).sum(1)
                            + (g2 * (p[:, i2] - p_prev[:, i2])).sum(1))
            dlam = np.where(denom > 0, (-C - damp) / np.maximum(denom, 1e-30), 0.0)
            p[:, i0] += (w[:, i0] * dlam)[:, None] * g0
            p[:, i1] += (w[:, i1] * dlam)[:, None] * g1
            p[:, i2] += (w[:, i2] * dlam)[:, None] * g2

    def _project_anchors(self, dt: float) -> None:
        """Zero-compliance attachment to the anchor point; the correction's
        force (|dp| / w / dt^2) is the grip force; above the limit the sucker
        pops and the node is free."""
        if not self.anchored.any():
            return
        p = self.p
        d = self.anchor_pos - p
        force = np.linalg.norm(d, axis=2) / np.maximum(self.w, 1e-30) / (dt * dt)
        self.anchor_force = np.where(self.anchored, force, 0.0)
        pop = self.anchored & (force > self.grip_limit)
        self.popped |= pop
        self.anchored &= ~pop
        p[self.anchored] = self.anchor_pos[self.anchored]

    def _project_body(self, dt: float, body: Body2D, root_local: np.ndarray) -> None:
        """Nodes 0 and 1 of every arm ride on the body: rigid attachment
        constraints, projected one after another (PBD rigid-point attachment).
        root_local: (A, 2, 2) the two root points in the body frame."""
        for a in range(self.A):
            for k in (0, 1):
                self._attach(a, k, body, root_local[a, k])

    def _attach(self, a: int, k: int, body: Body2D, local: np.ndarray) -> None:
        p, w = self.p, self.w
        R = body.rot()
        r = local @ R.T
        C = p[a, k] - (body.x + r)
        c = np.sqrt(C @ C)
        if c < 1e-12:
            return
        n = C / c
        rxn = r[0] * n[1] - r[1] * n[0]
        w_b = body.w + body.w_rot * rxn * rxn
        dlam = -c / (w[a, k] + w_b)
        p[a, k] += w[a, k] * dlam * n
        body.x -= body.w * dlam * n
        body.theta -= body.w_rot * rxn * dlam

    def tick(self, body: Body2D | None = None, root_local: np.ndarray | None = None,
             before_substep=None) -> None:
        dt = self.prm.dt_tick / self.prm.n_sub
        self.popped[:] = False
        for _ in range(self.prm.n_sub):
            if before_substep is not None:
                before_substep(self)
            self.substep(dt, body, root_local)


def segment_angles_bank(p: np.ndarray) -> np.ndarray:
    d = p[:, 1:] - p[:, :-1]
    d1, d2 = d[:, :-1], d[:, 1:]
    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    dot = (d1 * d2).sum(axis=2)
    return np.arctan2(cross, dot)
