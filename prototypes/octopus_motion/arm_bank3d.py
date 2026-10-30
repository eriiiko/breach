"""The 3-D arm bank: the 3-D chain (`arm_chain3d.py`) plus what the animal
needs -- rigid bodies with a 2-D pose plus height (x, y, z, heading; no pitch
or roll), the arm roots attached to them, sucker anchors with a grip limit,
and grabs (rigid attachments of arm nodes to another body, e.g. a prop).

Batched over EVERY arm of every animal in a scenario: several octopuses share
one bank (one `World`), so a gate's runs step together. The attachment order
is the 2-D bank's (for each body: arm 0 node 0, arm 0 node 1, arm 1 node 0,
...), vectorised over bodies, so batching never changes a result.

Each animal sees its own arms and body through `BankView` / `BodyView`, which
present the 2-D bank's interface (`anchor`, `release`, `activation[a]`,
`theta_target[a]` = the plane-B targets, `body.x` = the horizontal position, ...)
so the P2-P5 hubs run unchanged on top.

Author: Claude (octopus lead), 2026-10-30. Technique: XPBD (Macklin 2016),
Small Steps (Macklin 2019), PBD rigid-point attachment, Yekutieli 2005 I/II.
"""
from __future__ import annotations

import numpy as np

from arm_chain import N_NODES, N_SEG, ArmGeometry
from arm_chain3d import DRAG_LINEAR, DRAG_QUADRATIC, DRAG_UNIFORM, Chain3D, ChainParams3D

N_BEND = N_SEG - 1


class Bank3D(Chain3D):
    def __init__(self, geom: ArmGeometry, prm: ChainParams3D, rho_fluid: float | None = None):
        super().__init__(geom, prm)
        self.rho_fluid = geom.rho_water if rho_fluid is None else rho_fluid
        # bodies
        self.B = 0
        for name, shape, dtype, fill in self._body_fields():
            setattr(self, name, np.full((0,) + shape, fill, dtype=dtype))
        self.grabs: list[tuple[int, int, int, np.ndarray]] = []   # (arm, node, body, local (3,))
        self.views: list[BankView] = []
        self.record = None   # set to a list to record solver inputs per tick

    def _fields(self):
        N = N_NODES
        return super()._fields() + [
            ("anchored", (N,), bool, False), ("anchor_pos", (N, 3), float, 0.0),
            ("anchor_force", (N,), float, 0.0), ("grip", (), float, np.inf), ("popped", (N,), bool, False),
            ("arm_body", (), np.int64, -1), ("arm_slot", (), np.int64, -1), ("root_local", (2, 3), float, 0.0),
        ]

    @staticmethod
    def _body_fields():
        return [
            ("bx", (3,), float, 0.0), ("bv", (3,), float, 0.0), ("bth", (), float, 0.0), ("bom", (), float, 0.0),
            ("bm", (), float, 1.0), ("binertia", (), float, 1.0), ("bradius", (), float, 0.0),
            ("bcd", (), float, 1.0), ("barea", (), float, 0.0), ("brot_damp", (), float, 0.0),
            ("bdrag_mode", (), np.int64, DRAG_QUADRATIC), ("bvref", (), float, 0.0), ("btau", (), float, 1.0),
            ("bmu", (), float, 0.0), ("bgrav", (), bool, True), ("bfloor", (), bool, True),
        ]

    def add_body(self, x, theta=0.0, mass=1.0, radius=0.1, c_drag=1.0, area=0.0, rot_damp=0.0, mu=0.0,
                 inertia=None, gravity=True, floor=True) -> int:
        vals = dict(bx=np.asarray(x, float), bth=theta, bm=mass, bradius=radius, bcd=c_drag, barea=area,
                    brot_damp=rot_damp, bmu=mu, binertia=(0.5 * mass * radius ** 2 if inertia is None else inertia),
                    bgrav=gravity, bfloor=floor)
        for name, shape, dtype, fill in self._body_fields():
            arr = np.full((1,) + shape, fill, dtype=dtype)
            if name in vals:
                arr[0] = vals[name]
            setattr(self, name, np.concatenate([getattr(self, name), arr], axis=0))
        self.B += 1
        self._slot_cache = None
        return self.B - 1

    def add_arms(self, p0, pinned=(), body: int = -1, root_local=None, **per_arm) -> slice:
        n = np.asarray(p0).shape[0]
        extra = {}
        if body >= 0:
            extra["arm_body"] = body
            existing = int((self.arm_body == body).sum())
            extra_slots = np.arange(existing, existing + n)
        sl = super().add_arms(p0, pinned=pinned, **per_arm, **extra)
        if body >= 0:
            self.arm_slot[sl] = extra_slots
            self.root_local[sl] = root_local
        self._slot_cache = None
        return sl

    # -- anchors ---------------------------------------------------------------
    def anchor(self, arm: int, nodes, grip_limit: float | None = None) -> None:
        self.anchored[arm, nodes] = True
        self.anchor_pos[arm, nodes] = self.p[arm, nodes]
        if grip_limit is not None:
            self.grip[arm] = grip_limit

    def release(self, arm: int) -> None:
        self.anchored[arm, :] = False

    # -- body helpers -----------------------------------------------------------
    def _slots(self):
        """Per attachment slot s: (arm indices, body indices) -- one arm per body that has a slot s."""
        if self._slot_cache is None:
            out = []
            n_slots = int(self.arm_slot.max()) + 1 if self.A and (self.arm_slot >= 0).any() else 0
            for s in range(n_slots):
                arms = np.where(self.arm_slot == s)[0]
                out.append((arms, self.arm_body[arms]))
            self._slot_cache = out
        return self._slot_cache

    def _predict_bodies(self, dt: float) -> None:
        if self.B == 0:
            return
        self._bx_prev = self.bx.copy()
        self._bth_prev = self.bth.copy()
        v = self.bv
        sp = np.sqrt((v * v).sum(axis=1))
        k_q = 0.5 * self.rho_fluid * self.bcd * self.barea / self.bm
        f = np.where(self.bdrag_mode == DRAG_QUADRATIC, 1.0 / (1.0 + dt * k_q * sp),
                     np.where(self.bdrag_mode == DRAG_LINEAR, 1.0 / (1.0 + dt * k_q * self.bvref),
                              1.0 / (1.0 + dt / self.btau)))
        v *= f[:, None]
        if self.prm.g_eff:
            v[:, 2] -= dt * self.prm.g_eff * self.bgrav
        self.bom *= np.where(self.bdrag_mode == DRAG_UNIFORM, 1.0 / (1.0 + dt / self.btau),
                             1.0 / (1.0 + dt * self.brot_damp / self.binertia))
        self.bx += dt * v
        self.bth += dt * self.bom

    def _attach(self, arms, ks, bodies, local) -> None:
        """Rigid attachment of nodes (arms, ks) to body points `local` (n, 3), vectorised
        over distinct bodies; yaw-only rotation, so the angular term is the z part of r x n."""
        p, w = self.p, self.w
        th = self.bth[bodies]
        c, s = np.cos(th), np.sin(th)
        r = np.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1], local[:, 2]], axis=1)
        C = p[arms, ks] - (self.bx[bodies] + r)
        cn = np.sqrt((C * C).sum(axis=1))
        ok = cn > 1e-12
        n = C / np.maximum(cn, 1e-300)[:, None]
        rxn = r[:, 0] * n[:, 1] - r[:, 1] * n[:, 0]
        w_b = 1.0 / self.bm[bodies] + rxn * rxn / self.binertia[bodies]
        dlam = np.where(ok, -cn / (w[arms, ks] + w_b), 0.0)
        p[arms, ks] += (w[arms, ks] * dlam)[:, None] * n
        self.bx[bodies] -= (dlam / self.bm[bodies])[:, None] * n
        self.bth[bodies] -= rxn * dlam / self.binertia[bodies]

    def _project_extra(self, dt: float) -> None:
        # anchors: zero-compliance attachment to a world point; force |dp| / w / dt^2; pop above the grip
        if self.anchored.any():
            d = self.anchor_pos - self.p
            force = np.sqrt((d * d).sum(axis=2)) / np.maximum(self.w, 1e-30) / (dt * dt)
            self.anchor_force = np.where(self.anchored, force, 0.0)
            pop = self.anchored & (force > self.grip[:, None])
            self.popped |= pop
            self.anchored &= ~pop
            self.p[self.anchored] = self.anchor_pos[self.anchored]
        # the roots: for each body in order arm 0 node 0, arm 0 node 1, arm 1 node 0, ...
        for arms, bodies in self._slots():
            for k in (0, 1):
                self._attach(arms, np.full(len(arms), k), bodies, self.root_local[arms, k])
        for a, k, b, local in self.grabs:
            self._attach(np.array([a]), np.array([k]), np.array([b]), local[None, :])

    def _floor_bodies(self, p_prev) -> None:
        if self.B == 0 or not self.floor_on:
            return
        z = self.bx[:, 2]
        pen = np.where(self.bfloor & (z < 0.0), -z, 0.0)
        if not (pen > 0).any():
            return
        z += pen
        dx = self.bx[:, :2] - self._bx_prev[:, :2]
        s = np.sqrt((dx * dx).sum(axis=1))
        keep = np.where(pen > 0, np.clip(1.0 - self.bmu * pen / np.maximum(s, 1e-300), 0.0, 1.0), 1.0)
        self.bx[:, :2] = self._bx_prev[:, :2] + dx * keep[:, None]

    def _before_predict(self, dt: float) -> None:
        self.v[self.anchored] = 0.0

    def _after_velocity(self, dt: float) -> None:
        self.v[self.anchored] = 0.0
        if self.B:
            self.bv[:] = (self.bx - self._bx_prev) / dt
            if self.floor_on:
                on = self.bfloor & (self.bx[:, 2] <= 1e-12)
                self.bv[:, 2] = np.where(on, np.maximum(self.bv[:, 2], 0.0), self.bv[:, 2])
            self.bom[:] = (self.bth - self._bth_prev) / dt

    def tick(self) -> None:
        self.popped[:] = False
        super().tick()

    # -- the solver's per-tick inputs, for the integer reference's replay ------
    def snapshot_inputs(self) -> dict:
        grab_body = np.full((self.A, N_NODES), -1, dtype=np.int64)
        grab_local = np.zeros((self.A, N_NODES, 3))
        for a, k, b, local in self.grabs:
            grab_body[a, k] = b
            grab_local[a, k] = local
        return dict(activation=self.activation.copy(), rest_scale=self.rest_scale.copy(),
                    bend_target=self.bend_target.copy(), anchored=self.anchored.copy(),
                    anchor_pos=self.anchor_pos.copy(), root_local=self.root_local.copy(), grip=self.grip.copy(),
                    grab_body=grab_body, grab_local=grab_local, bv=self.bv.copy(), bom=self.bom.copy())


# ------------------------------------------------------------ views for the 2-D hubs

class BodyView:
    """One bank body seen through the 2-D Body2D interface (x, v are the
    horizontal parts, as numpy views; z separately)."""

    def __init__(self, bank: Bank3D, index: int):
        self.bank, self.i = bank, index

    x = property(lambda s: s.bank.bx[s.i, :2])
    v = property(lambda s: s.bank.bv[s.i, :2])
    x3 = property(lambda s: s.bank.bx[s.i])
    v3 = property(lambda s: s.bank.bv[s.i])
    radius = property(lambda s: float(s.bank.bradius[s.i]))
    mass = property(lambda s: float(s.bank.bm[s.i]))
    w = property(lambda s: 1.0 / float(s.bank.bm[s.i]))

    @x.setter
    def x(self, val):
        self.bank.bx[self.i, :2] = val

    @v.setter
    def v(self, val):
        self.bank.bv[self.i, :2] = val

    @property
    def z(self) -> float:
        return float(self.bank.bx[self.i, 2])

    @property
    def theta(self) -> float:
        return float(self.bank.bth[self.i])

    @theta.setter
    def theta(self, val):
        self.bank.bth[self.i] = val

    @property
    def omega(self) -> float:
        return float(self.bank.bom[self.i])

    def rot(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def rot3(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class BankView:
    """The arms [a0, a0 + n) of a bank, through the 2-D ArmBank interface."""

    def __init__(self, bank: Bank3D, sl: slice):
        self.bank, self.sl = bank, sl
        self.a0 = sl.start
        self.grabs: list[tuple[int, int, BodyView, np.ndarray]] = []   # local arm index
        bank.views.append(self)

    def _arr(name):
        return property(lambda s: getattr(s.bank, name)[s.sl])

    p = _arr("p")
    v = _arr("v")
    m = _arr("m")
    w = _arr("w")
    activation = _arr("activation")
    rest_scale = _arr("rest_scale")
    rest0 = _arr("rest0")
    anchored = _arr("anchored")
    anchor_pos = _arr("anchor_pos")
    anchor_force = _arr("anchor_force")
    popped = _arr("popped")
    alive = _arr("alive")
    bend_target = _arr("bend_target")
    K_per = _arr("K_per")
    K_tan = _arr("K_tan")
    del _arr

    @property
    def theta_target(self):
        """The plane-B targets (counter-clockwise seen from above for a deck-lying arm) -- the 2-D hubs' theta_target."""
        return self.bank.bend_target[self.sl, :, 1]

    @property
    def rest(self):
        return self.bank.rest0[self.sl] * self.bank.rest_scale[self.sl]

    @property
    def s_node(self):
        return self.bank.s_node

    @property
    def grip_limit(self):
        return self.bank.grip[self.sl]

    @grip_limit.setter
    def grip_limit(self, val):
        self.bank.grip[self.sl] = val

    def anchor(self, arm: int, nodes, grip_limit: float | None = None) -> None:
        self.bank.anchor(self.a0 + arm, nodes, grip_limit)

    def release(self, arm: int) -> None:
        self.bank.release(self.a0 + arm)

    def add_grab(self, arm: int, node: int, body: BodyView, local3: np.ndarray) -> None:
        self.grabs.append((arm, node, body, local3))
        self.bank.grabs.append((self.a0 + arm, node, body.i, np.asarray(local3, float)))


class World:
    """Animals that step together on one bank. Each animal implements pre_tick()
    (its hub writes this tick's commands) and post_tick() (it logs)."""

    def __init__(self, geom: ArmGeometry, prm: ChainParams3D, rho_fluid: float | None = None):
        self.bank = Bank3D(geom, prm, rho_fluid)
        self.animals = []
        self.time = 0.0
        self.recorders = []   # callables(world, phase) for traces

    def add(self, animal) -> None:
        self.animals.append(animal)

    def tick(self) -> None:
        for a in self.animals:
            a.pre_tick()
        for r in self.recorders:
            r(self, "inputs")
        self.bank.tick()
        self.time += self.bank.prm.dt_tick
        for a in self.animals:
            a.post_tick()
        for r in self.recorders:
            r(self, "state")

    def run(self, seconds: float, until=None) -> None:
        n = int(round(seconds / self.bank.prm.dt_tick))
        for _ in range(n):
            self.tick()
            if until is not None and until():
                break
