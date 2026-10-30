"""The octopus arm chain in 3-D, batched over arms. Prototype P5z (float, game scale).

This is the P1 chain (`arm_chain.py`) lifted to three dimensions, constraint for
constraint, with what the game needs on top: a floor, buoyancy-reduced gravity
and two signed bends per interior node about a parallel-transported (Bishop)
frame. It is the float specification the integer reference
(`docs/octopus_chain_ref_q.py`, P6b) is written against and compared to.

State per arm: N = 18 nodes, positions p (A, N, 3) and velocities v (A, N, 3)
in metres and m/s; z is UP and the floor (the deck) is the plane z = 0.

One substep (dt = dt_tick / n_sub), in this order -- every step is what the
port carries:

  1. activation   per-arm stiffening wave, Yekutieli's sigmoid
                  a = a_ca * S((V (t - t0) - s_i) / w),  S(x) = (1 + tanh x) / 2,
                  evaluated EITHER with tanh OR through the tabulated S
                  (`ACT_TABLE`: 1025 samples on x in [-8, 8], step 1/64, linear
                  interpolation, clamped outside) -- the table is what the port uses.
  2. drag         per node on its own tangent t (central difference), implicit
                  per part so it can never overshoot:
                  v_t <- v_t / (1 + dt K_tan |v_t| w),  v_n <- v_n / (1 + dt K_per |v_n| w)
                  with v_n the 3-D normal part (a vector).  Two CONTROL laws:
                  "linear" (isotropic, K_per * v_ref on both parts) and "uniform"
                  (v <- v / (1 + dt / tau) on every node: mass-proportional, the
                  law under which no shape change can move the centre of mass).
  3. gravity      v_z -= dt g_eff on every free node (g_eff = g (1 - rho_w / rho_arm)).
  4. predict      p_prev = p;  p += dt v.
  5. n_pass Gauss-Seidel passes, each:
       a. distance  17 constraints |p_{i+1} - p_i| = rest_i * rest_scale_i,
                    dlam = -C / (w_i + w_{i+1} + 1/(k_dist dt^2))   (lambda reset per pass)
       b. frames    the Bishop frame of every segment, from the CURRENT positions
                    (`bishop_frames`, documented there), frozen for the bend sweep
       c. bend      16 interior nodes x 2 planes, base -> tip, plane A then plane B
                    at each node: the 2-D P1 constraint in that plane
                    (`_project_bend_plane`), lambda reset per pass
       d. (bank)    anchors, body attachments, grabs  (arm_bank3d.py)
       e. floor     one-sided: z < 0 -> z = 0, and Coulomb friction on the
                    substep's tangential displacement:
                    |dx_t| <- max(0, |dx_t| - mu * penetration)
  6. velocities   v = (p - p_prev) / dt; pinned nodes 0; a node on the floor
                  keeps no downward velocity (v_z <- max(v_z, 0)).

Author: Claude (octopus lead), 2026-10-30. Technique: Macklin, Mueller,
Chentanez 2016 "XPBD" (10.1145/2994258.2994272); Macklin et al. 2019 "Small
steps" (10.1145/3309486.3340247); Bishop 1975 "There is more than one way to
frame a curve" (Amer. Math. Monthly 82:246) for the parallel-transported frame;
the activation wave and the drag law from Yekutieli et al. 2005 I/II
(10.1152/jn.00684.2004, 10.1152/jn.00685.2004); position-based friction after
Macklin et al. 2014 "Unified particle physics" (10.1145/2601097.2601152).
PDFs: docs/papers/octopus/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_chain import N_NODES, N_SEG, ArmGeometry

N_BEND = N_SEG - 1

DRAG_QUADRATIC, DRAG_LINEAR, DRAG_UNIFORM = 0, 1, 2

# ------------------------------------------------------------ the activation table
# S(x) = (1 + tanh x) / 2 sampled at x_i = -8 + i / 64, i = 0..1024 (1025 samples, 1024 intervals).
# The step is a power of two so the port's index is a shift of the Q16 argument; outside [-8, 8]
# the value is clamped (S(-8) = 1.1e-7, S(8) = 1 - 1.1e-7). Linear interpolation error <= 6e-6.
ACT_X_MAX = 8.0
ACT_INTERVALS = 1024
ACT_STEP = 2.0 * ACT_X_MAX / ACT_INTERVALS            # 1/64
ACT_TABLE = 0.5 * (1.0 + np.tanh(np.linspace(-ACT_X_MAX, ACT_X_MAX, ACT_INTERVALS + 1)))


def sigmoid_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(x))


def sigmoid_table(x: np.ndarray) -> np.ndarray:
    """The tabulated S: clamp to [-8, 8], index = floor((x + 8) * 64), linear interpolation."""
    u = (np.clip(x, -ACT_X_MAX, ACT_X_MAX) + ACT_X_MAX) / ACT_STEP
    i = np.minimum(np.floor(u).astype(np.int64), ACT_INTERVALS - 1)
    f = u - i
    return ACT_TABLE[i] * (1.0 - f) + ACT_TABLE[i + 1] * f


@dataclass(frozen=True)
class ChainParams3D:
    """Everything the 3-D solver needs besides the geometry. Units SI."""
    dt_tick: float = 1.0 / 24.0
    n_sub: int = 16
    n_pass: int = 2
    k_bend: float = 0.1            # N m/rad, fully active, at the base
    k_passive_frac: float = 0.01
    k_dist: float = 2000.0         # N/m
    bend_damp_s: float = 1e-3      # s, Rayleigh damping time of the bend constraints
    c_per: float = 1.013
    c_tan: float = 0.0256
    g_eff: float = 0.0             # m/s^2 along -z
    activation_table: bool = False  # True: S(x) through ACT_TABLE (the port's form)


# ------------------------------------------------------------ the Bishop frame

def bishop_frames(p: np.ndarray, ref: np.ndarray, guard: float = 1e-9):
    """Parallel-transported frame of every segment of every arm.

    p: (A, N, 3) positions; ref: (A, 3) the root's reference direction (the
    body's UP for an arm on an animal -- the body has no pitch or roll, so this
    is world +z; a free arm carries its own).

    Returns t, nA, nB, each (A, N_SEG, 3), orthonormal per segment:

      t_j   = (p_{j+1} - p_j) / |p_{j+1} - p_j|                  the tangent
      nA_0  = normalize(ref - (ref . t_0) t_0)                    the root normal
              (if |ref - (ref . t_0) t_0| < 1e-3 the arm points along ref and
              ref is replaced by the first of x, y not parallel to t_0)
      nA_{j+1} = nA_j - ((nA_j . t_{j+1}) / (1 + t_j . t_{j+1})) (t_j + t_{j+1})
              -- the exact minimal rotation taking t_j to t_{j+1} applied to nA_j
              (Rodrigues' rotation about t_j x t_{j+1} by the angle between
              them, written without trig: one division per segment); 1 + c is
              floored at `guard` (a 180-degree fold between two segments, which
              no shipped pose reaches);
              then re-orthonormalised: nA <- normalize(nA - (nA . t) t)  (a
              no-op in exact arithmetic; it stops rounding drift in the port)
      nB_j  = nA_j x t_j
              With nA = up and t along +x, nB = +y: a positive turn about
              plane B is COUNTER-CLOCKWISE SEEN FROM ABOVE, the 2-D prototype's
              sign, so a deck-lying arm's planar targets are plane B's.

    A straight arm transports the frame unchanged (c = 1, the correction is 0)
    -- no singularity at straight, which is the state a reach drives to.
    Twist is not represented: the frame is rotation-minimizing by definition.
    """
    d = p[:, 1:] - p[:, :-1]
    ln = np.sqrt((d * d).sum(axis=2))
    t = d / np.maximum(ln, 1e-300)[..., None]
    A = p.shape[0]
    nA = np.empty((A, N_SEG, 3))
    t0 = t[:, 0]
    n0 = ref - (ref * t0).sum(axis=1)[:, None] * t0
    nn = np.sqrt((n0 * n0).sum(axis=1))
    bad = nn < 1e-3
    if bad.any():
        alt = np.where(np.abs(t0[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
        n_alt = alt - (alt * t0).sum(axis=1)[:, None] * t0
        n0 = np.where(bad[:, None], n_alt, n0)
        nn = np.sqrt((n0 * n0).sum(axis=1))
    nA[:, 0] = n0 / nn[:, None]
    for j in range(N_SEG - 1):
        tj, tk = t[:, j], t[:, j + 1]
        n = nA[:, j]
        c = (tj * tk).sum(axis=1)
        f = (n * tk).sum(axis=1) / np.maximum(1.0 + c, guard)
        n = n - f[:, None] * (tj + tk)
        n = n - (n * tk).sum(axis=1)[:, None] * tk
        nA[:, j + 1] = n / np.sqrt((n * n).sum(axis=1))[:, None]
    nB = np.cross(nA, t)
    return t, nA, nB


def bend_angles(p: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(A, N_BEND, 2) the two signed turns at every interior node: the angle
    from segment j to segment j+1 in plane A (spanned by t_j, nA_j) and in plane
    B (t_j, nB_j) -- exactly what the bend constraints measure."""
    t, nA, nB = bishop_frames(p, ref)
    d = p[:, 1:] - p[:, :-1]
    d1, d2 = d[:, :-1], d[:, 1:]
    e1 = t[:, :-1]
    out = np.empty(p.shape[:1] + (N_BEND, 2))
    a1x = (d1 * e1).sum(2)
    a2x = (d2 * e1).sum(2)
    for k, eX in enumerate((nA[:, :-1], nB[:, :-1])):
        a1y = (d1 * eX).sum(2)
        a2y = (d2 * eX).sum(2)
        out[..., k] = np.arctan2(a1x * a2y - a1y * a2x, a1x * a2x + a1y * a2y)
    return out


# ------------------------------------------------------------ the chain

class Chain3D:
    """A batch of independent arms (A may be 1). Subclassed by the bank, which
    adds anchors, bodies and grabs; this class owns the arm physics."""

    def __init__(self, geom: ArmGeometry, prm: ChainParams3D):
        self.geom = geom
        self.prm = prm
        self.A = 0
        r = geom.radii()
        self.radii = r
        self.rest_nominal = geom.rest_lengths()
        self.s_node = np.concatenate([[0.0], np.cumsum(self.rest_nominal)])
        r_ratio = r[1:-1] / geom.r_base_m
        self.k_active = prm.k_bend * r_ratio ** 3
        self.k_passive = prm.k_passive_frac * prm.k_bend * r_ratio ** 3
        l_node = np.zeros(N_NODES)
        l_node[:-1] += 0.5 * self.rest_nominal
        l_node[1:] += 0.5 * self.rest_nominal
        self._K_per0 = 0.5 * geom.rho_water * (2.0 * r * l_node) * prm.c_per
        self._K_tan0 = 0.5 * geom.rho_water * (2.0 * np.pi * r * l_node) * prm.c_tan
        self._m0 = geom.masses()
        self.time = 0.0
        self.sigmoid = sigmoid_table if prm.activation_table else sigmoid_tanh
        self.floor_on = True
        for name, shape, dtype, fill in self._fields():
            setattr(self, name, np.full((0,) + shape, fill, dtype=dtype))

    # per-arm state, grown by add_arms (name, shape per arm, dtype, default)
    def _fields(self):
        N = N_NODES
        return [
            ("p", (N, 3), float, 0.0), ("v", (N, 3), float, 0.0),
            ("m", (N,), float, 1.0), ("w", (N,), float, 1.0),
            ("rest0", (N_SEG,), float, 0.0), ("rest_scale", (N_SEG,), float, 1.0),
            ("activation", (N_BEND,), float, 0.0), ("bend_target", (N_BEND, 2), float, 0.0),
            ("K_per", (N,), float, 0.0), ("K_tan", (N,), float, 0.0),
            ("drag_mode", (), np.int64, DRAG_QUADRATIC), ("drag_vref", (), float, 0.0), ("drag_tau", (), float, 1.0),
            ("mu", (), float, 0.0), ("root_ref", (3,), float, 0.0), ("alive", (), bool, True),
            ("pinned", (N,), bool, False),
            ("wave_on", (), bool, False), ("wave_a", (), float, 0.0), ("wave_tact", (), float, 1.0),
            ("wave_t0", (), float, 0.0), ("wave_w", (), float, 1.0), ("wave_hold", (), bool, True),
        ]

    def add_arms(self, p0: np.ndarray, pinned=(), **per_arm) -> slice:
        """Append arms with initial positions p0 (n, N, 3). Returns their slice."""
        p0 = np.asarray(p0, dtype=float)
        n = p0.shape[0]
        new = {}
        for name, shape, dtype, fill in self._fields():
            new[name] = np.full((n,) + shape, fill, dtype=dtype)
        new["p"][:] = p0
        new["m"][:] = self._m0
        new["rest0"][:] = self.rest_nominal
        new["K_per"][:] = self._K_per0
        new["K_tan"][:] = self._K_tan0
        new["root_ref"][:] = (0.0, 0.0, 1.0)
        new["wave_w"][:] = self.geom.length_m / N_SEG
        new["pinned"][:, list(pinned)] = True
        for k, val in per_arm.items():
            new[k][:] = val
        new["w"][:] = np.where(new["pinned"], 0.0, 1.0 / new["m"])
        for name in new:
            setattr(self, name, np.concatenate([getattr(self, name), new[name]], axis=0))
        sl = slice(self.A, self.A + n)
        self.A += n
        return sl

    # -- activation --------------------------------------------------------
    def set_waves(self, sl, a_ca, t_act, t_start=0.0, width=None, hold=True) -> None:
        self.wave_on[sl] = True
        self.wave_a[sl] = a_ca
        self.wave_tact[sl] = t_act
        self.wave_t0[sl] = t_start
        if width is not None:
            self.wave_w[sl] = width
        self.wave_hold[sl] = hold

    def _apply_waves(self) -> None:
        on = self.wave_on
        if not on.any():
            return
        L = self.geom.length_m
        s = self.s_node[1:-1][None, :]
        s_front = (L / self.wave_tact[on] * (self.time - self.wave_t0[on]))[:, None]
        w = self.wave_w[on][:, None]
        a = self.sigmoid((s_front - s) / w)
        hold = self.wave_hold[on]
        if not hold.all():
            fade = self.sigmoid((s - (s_front - 0.4 * L)) / w)
            a = np.where(hold[:, None], a, a * fade)
        self.activation[on] = self.wave_a[on][:, None] * a

    # -- drag ----------------------------------------------------------------
    def _drag(self, dt: float) -> None:
        p, v, w = self.p, self.v, self.w
        tang = np.empty_like(p)
        tang[:, 1:-1] = p[:, 2:] - p[:, :-2]
        tang[:, 0] = p[:, 1] - p[:, 0]
        tang[:, -1] = p[:, -1] - p[:, -2]
        tang /= np.maximum(np.sqrt((tang * tang).sum(axis=2)), 1e-300)[..., None]
        v_t = (v * tang).sum(axis=2)
        vt_vec = v_t[..., None] * tang
        v_n = v - vt_vec
        mode = self.drag_mode
        if (mode == DRAG_QUADRATIC).all():
            sn = np.sqrt((v_n * v_n).sum(axis=2))
            ft = 1.0 / (1.0 + dt * self.K_tan * np.abs(v_t) * w)
            fn = 1.0 / (1.0 + dt * self.K_per * sn * w)
        else:
            sn = np.sqrt((v_n * v_n).sum(axis=2))
            ft_q = 1.0 / (1.0 + dt * self.K_tan * np.abs(v_t) * w)
            fn_q = 1.0 / (1.0 + dt * self.K_per * sn * w)
            f_lin = 1.0 / (1.0 + dt * self.K_per * self.drag_vref[:, None] * w)
            f_uni = np.broadcast_to((1.0 / (1.0 + dt / self.drag_tau))[:, None], ft_q.shape)
            sel = mode[:, None]
            ft = np.where(sel == DRAG_QUADRATIC, ft_q, np.where(sel == DRAG_LINEAR, f_lin, f_uni))
            fn = np.where(sel == DRAG_QUADRATIC, fn_q, np.where(sel == DRAG_LINEAR, f_lin, f_uni))
        v[:] = (ft * v_t)[..., None] * tang + fn[..., None] * v_n

    # -- constraints ------------------------------------------------------------
    def _project_distance(self, dt: float) -> None:
        p, w = self.p, self.w
        alpha = (1.0 / self.prm.k_dist) / (dt * dt)
        rest = self.rest0 * self.rest_scale
        for i in range(N_SEG):
            d = p[:, i + 1] - p[:, i]
            ln = np.maximum(np.sqrt((d * d).sum(axis=1)), 1e-12)
            C = ln - rest[:, i]
            w0, w1 = w[:, i], w[:, i + 1]
            dlam = -C / (w0 + w1 + alpha)
            g = d / ln[:, None]
            p[:, i] -= (w0 * dlam)[:, None] * g
            p[:, i + 1] += (w1 * dlam)[:, None] * g

    def _project_bend(self, dt: float, p_prev: np.ndarray) -> None:
        """Bend sweep base -> tip; at each interior node plane A then plane B.

        At node j+1, plane X in (A, B) with axes (e1, eX) = (t_j, nX_j) of the
        frame computed at the start of this sweep (frozen during it):
          a1 = (d1 . e1, d1 . eX),  a2 = (d2 . e1, d2 . eX)      in-plane parts
          theta = atan2(a1 x a2, a1 . a2),  C = theta - target_X
          g0 = perp(a1) / |a1|^2,  g2 = perp(a2) / |a2|^2,  g1 = -g0 - g2,
          perp(x, y) = (-y, x)  -- the P1 gradient, lifted G_i = g_ix e1 + g_iy eX
          W = sum w_i |g_i|^2,  D = gamma sum G_i . (p_i - p_prev_i),  gamma = bend_damp_s / dt
          dlam = (-C - D) / ((1 + gamma) W + 1 / (k dt^2)),   k = k_passive + act * k_active
          p_i += w_i dlam G_i
        Complex numbers below are only numpy shorthand for the 2-D in-plane
        vectors (a = ax + i ay; perp(a) = i a; a1 x a2 + i a1 . a2 = i conj(a1) a2).
        """
        t, nA, nB = bishop_frames(self.p, self.root_ref)
        # basis[:, j, plane] = (e1, eX) rows, (A, N_BEND, 2, 2, 3)
        basis = np.stack([np.stack([t[:, :-1], nA[:, :-1]], axis=2),
                          np.stack([t[:, :-1], nB[:, :-1]], axis=2)], axis=2)
        basisT = np.swapaxes(basis, -1, -2)
        gamma = self.prm.bend_damp_s / dt
        k_all = self.k_passive[None, :] + self.activation * self.k_active[None, :]
        alpha_all = (1.0 / (dt * dt)) / k_all
        p, w = self.p, self.w
        A = self.A
        G = np.empty((A, 3), dtype=np.complex128)
        for j in range(N_BEND):
            w3 = w[:, j:j + 3]
            alpha = alpha_all[:, j]
            pp3 = p_prev[:, j:j + 3]
            for plane in (0, 1):
                P3 = p[:, j:j + 3]
                D = np.concatenate([P3[:, 1:] - P3[:, :-1], P3 - pp3], axis=1)   # d1, d2, q0, q1, q2
                z = (D @ basisT[:, j, plane]).view(np.complex128)[..., 0]      # (A, 5) in-plane coordinates
                z1, z2 = z[:, 0], z[:, 1]
                pr = np.conj(z1) * z2
                C = np.arctan2(pr.imag, pr.real) - self.bend_target[:, j, plane]
                g0 = 1j / np.conj(z1)                              # perp(a1) / |a1|^2
                g2 = 1j / np.conj(z2)
                G[:, 0] = g0
                G[:, 1] = -g0 - g2
                G[:, 2] = g2
                Gr, Gi = G.real, G.imag
                Wsum = (w3 * (Gr * Gr + Gi * Gi)).sum(axis=1)
                zq = z[:, 2:]
                damp = gamma * (Gr * zq.real + Gi * zq.imag).sum(axis=1)
                dlam = (-C - damp) / ((1.0 + gamma) * Wsum + alpha)
                c = (w3 * dlam[:, None]) * G
                p[:, j:j + 3] += c.view(np.float64).reshape(A, 3, 2) @ basis[:, j, plane]

    def _project_floor(self, p_prev: np.ndarray) -> None:
        if not self.floor_on:
            return
        p = self.p
        z = p[..., 2]
        pen = np.where((z < 0.0) & ~self.pinned, -z, 0.0)
        if not (pen > 0).any():
            return
        z += pen
        mu = self.mu[:, None]
        if (mu > 0).any():
            dx = p[..., :2] - p_prev[..., :2]
            s = np.sqrt((dx * dx).sum(axis=2))
            keep = np.where(pen > 0, np.clip(1.0 - mu * pen / np.maximum(s, 1e-300), 0.0, 1.0), 1.0)
            p[..., :2] = p_prev[..., :2] + dx * keep[..., None]

    # -- hooks for the bank ------------------------------------------------------
    def _before_predict(self, dt: float) -> None:
        pass

    def _predict_bodies(self, dt: float) -> None:
        pass

    def _project_extra(self, dt: float) -> None:
        pass

    def _after_velocity(self, dt: float) -> None:
        pass

    # -- one substep -----------------------------------------------------------
    def substep(self, dt: float) -> None:
        self._apply_waves()
        self._drag(dt)
        if self.prm.g_eff:
            self.v[..., 2] -= dt * self.prm.g_eff * (self.w > 0)
        self.v[self.pinned] = 0.0
        self._before_predict(dt)
        p_prev = self.p.copy()
        self.p += dt * self.v
        self._predict_bodies(dt)
        for _ in range(self.prm.n_pass):
            self._project_distance(dt)
            self._project_bend(dt, p_prev)
            self._project_extra(dt)
            self._project_floor(p_prev)
            self._floor_bodies(p_prev)
        self.v[:] = (self.p - p_prev) / dt
        self.v[self.pinned] = 0.0
        if self.floor_on:
            on_floor = self.p[..., 2] <= 1e-12
            self.v[..., 2] = np.where(on_floor, np.maximum(self.v[..., 2], 0.0), self.v[..., 2])
        self._after_velocity(dt)
        self.time += dt

    def _floor_bodies(self, p_prev) -> None:
        pass

    def tick(self) -> None:
        dt = self.prm.dt_tick / self.prm.n_sub
        for _ in range(self.prm.n_sub):
            self.substep(dt)


# ------------------------------------------------------------ poses

def planar_to_3d(p2: np.ndarray, origin=(0.0, 0.0, 0.0), ex=(1.0, 0.0, 0.0), ey=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Embed a 2-D pose (N, 2) in the plane spanned by ex, ey through origin."""
    return np.asarray(origin, float)[None, :] + p2[:, :1] * np.asarray(ex, float)[None, :] \
        + p2[:, 1:2] * np.asarray(ey, float)[None, :]


def rot_matrix(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
