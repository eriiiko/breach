"""One octopus arm as an XPBD particle chain, planar, pure numpy. Prototype P1.

The question this prototype answers (study brief v1 §5, issue #105): does a
single centre-line chain of 18 nodes, actuated by a STIFFENING WAVE that runs
base -> tip at constant speed, reproduce the real octopus reach that
Yekutieli et al. 2005 (J Neurophysiol 94:1443 / 94:1459) got from their 2D
dorsal/ventral mass-spring model -- a travelling bend with a bell-shaped
velocity, a straight stiff arm behind it, no tip overshoot, and kinematics
that do not change with the force level above a threshold?

Model (every piece is what the sim port would carry, so nothing here needs
more than +, -, *, one reciprocal per constraint, one atan2 per bend):

  nodes     N = 18 point masses along the arm, tapered (rho * pi R^2 * l),
            node 0 and node 1 pinned (the base and its direction = the aim).
  distance  17 XPBD constraints |p_{i+1} - p_i| = rest_i (rest length is the
            muscle's stretch / shorten dial; fixed here).
  bend      16 XPBD angle constraints theta_i = theta_target_i at each
            interior node, compliance 1 / k_i(t) where
            k_i(t) = k_passive_i + a_i(t) * k_active_i,
            k_active_i = k_bend * (R_i / R_base)^3   (torque ~ area * lever)
            a_i(t) = a_ca * 1/2 (1 + tanh((V t - s_i) / w)) -- Yekutieli's
            sigmoid front travelling at V = L / t_act; target angle 0.
  drag      quadratic, anisotropic, per node on its own tangent:
            F_n = -1/2 rho (2 R l) c_per |v_n| v_n,  F_t = -1/2 rho (2 pi R l)
            c_tan |v_t| v_t (Yekutieli I, measured on a dropped arm), applied
            as v *= 1 / (1 + dt K |v| / m) per component -- never overshoots.
  gravity   g (1 - rho_water / rho_arm) along -y (nearly neutral; a dead arm
            sinks slowly). Set g_eff = 0 for a horizontal reach plane.
  step      N_SUB substeps per tick x N_PASS Gauss-Seidel passes (Macklin 2019,
            "small steps"), Rayleigh damping on the bend constraints (XPBD).

Author: Claude (octopus lead), 2026-10-29. Technique: Macklin, Mueller,
Chentanez 2016 "XPBD" (10.1145/2994258.2994272); Macklin et al. 2019 "Small
steps in physics simulation" (10.1145/3309486.3340247); the activation wave
and the drag law from Yekutieli et al. 2005 I/II (10.1152/jn.00684.2004,
10.1152/jn.00685.2004). PDFs: docs/papers/octopus/.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- geometry

N_NODES = 18          # one per game bone + the tip (CONTROLS.md: 17 bones per arm)
N_SEG = N_NODES - 1


@dataclass(frozen=True)
class ArmGeometry:
    """A real Octopus vulgaris arm, Yekutieli I's numbers."""
    length_m: float = 0.24
    r_base_m: float = 0.0075      # 15 mm wide at the base
    r_tip_m: float = 0.0005       # 1 mm at the tip
    rho_arm: float = 1042.0       # kg/m^3
    rho_water: float = 1022.0

    def radii(self) -> np.ndarray:
        """Radius at each node, linear taper base -> tip."""
        return np.linspace(self.r_base_m, self.r_tip_m, N_NODES)

    def rest_lengths(self) -> np.ndarray:
        return np.full(N_SEG, self.length_m / N_SEG)

    def masses(self) -> np.ndarray:
        """Each node owns half of each adjacent segment's volume."""
        r = self.radii()
        l = self.rest_lengths()
        seg_vol = np.pi * (0.5 * (r[:-1] + r[1:])) ** 2 * l
        m = np.zeros(N_NODES)
        m[:-1] += 0.5 * seg_vol
        m[1:] += 0.5 * seg_vol
        return self.rho_arm * m


@dataclass(frozen=True)
class ChainParams:
    """Everything the solver needs besides the geometry. Units SI."""
    dt_tick: float = 1.0 / 24.0   # the game's [clock] ticks_per_second
    n_sub: int = 16               # substeps per tick (P1 finding: 8 x 1 leaves the arm behind the
    n_pass: int = 2               #   bend 6-7 deg curved; 16 x 2 resolves the stiffness, see report)
    k_bend: float = 0.1           # N*m/rad, fully active bend stiffness at the base
                                  # (Yekutieli I: 8e4 N/m^2 active stress x pi R^2 x lever R at R = 7.5 mm)
    k_passive_frac: float = 0.01  # passive bend stiffness as a fraction of k_bend at the base
    k_dist: float = 2000.0        # N/m, longitudinal stiffness (stiff, not rigid)
    bend_damp_s: float = 1e-3     # Rayleigh damping time constant of the bend constraints (s):
                                  # XPBD gamma = bend_damp_s / dt; Yekutieli's muscle has ~7e-5 s
    c_per: float = 1.013          # Yekutieli I, measured
    c_tan: float = 0.0256
    g_eff: float = 0.0            # m/s^2 along -y; 9.81*(1-1022/1042) = 0.188 for a vertical plane


@dataclass(frozen=True)
class ReachCommand:
    """The brain's whole say over a reach (Yekutieli II): aim is the pinned
    base direction, then amplitude + travel time."""
    a_ca: float = 0.6             # activation amplitude, 0..1
    t_act: float = 0.9            # s for the front to travel base -> tip
    t_start: float = 0.0          # s, when the wave leaves the base
    front_width_m: float | None = None   # sigmoid width; default one segment
    hold: bool = True             # keep activation behind the front (tonic)


# ----------------------------------------------------------------- posture

def curled_pose(geom: ArmGeometry, total_rad: float = 4.0, base_straight_frac: float = 0.15,
                base_angle: float = 0.0) -> np.ndarray:
    """A resting curl: straight for the first part, then curvature rising
    toward the tip (the model's photo: the arm curled dorsally near the body).
    `total_rad` = the total bend from base to tip."""
    rest = geom.rest_lengths()
    s_mid = np.cumsum(rest) - 0.5 * rest            # arc length at each segment centre
    s = s_mid / geom.length_m
    w = np.clip((s - base_straight_frac) / (1.0 - base_straight_frac), 0.0, None)
    w = w ** 1.5
    turn = total_rad * w / max(w.sum(), 1e-12)      # turn per segment, sums to total_rad
    ang = base_angle + np.concatenate([[0.0], np.cumsum(turn)[:-1]])
    pts = np.zeros((N_NODES, 2))
    for i in range(N_SEG):
        pts[i + 1] = pts[i] + rest[i] * np.array([np.cos(ang[i]), np.sin(ang[i])])
    return pts


def hooked_pose(geom: ArmGeometry, total_rad: float = 2.6, center_frac: float = 0.3,
                width_frac: float = 0.12, base_angle: float = 0.0) -> np.ndarray:
    """Yekutieli's natural start: the arm curled dorsally NEAR THE BODY -- one
    bend of `total_rad` centred at `center_frac` of the length (a Gaussian
    turn distribution of `width_frac`), the distal arm lying back nearly
    straight. The wave must push this bend out to the tip."""
    rest = geom.rest_lengths()
    s_mid = (np.cumsum(rest) - 0.5 * rest) / geom.length_m
    w = np.exp(-0.5 * ((s_mid - center_frac) / width_frac) ** 2)
    turn = total_rad * w / w.sum()
    ang = base_angle + np.concatenate([[0.0], np.cumsum(turn)[:-1]])
    pts = np.zeros((N_NODES, 2))
    for i in range(N_SEG):
        pts[i + 1] = pts[i] + rest[i] * np.array([np.cos(ang[i]), np.sin(ang[i])])
    return pts


def straight_pose(geom: ArmGeometry, base_angle: float = 0.0) -> np.ndarray:
    rest = geom.rest_lengths()
    d = np.array([np.cos(base_angle), np.sin(base_angle)])
    return np.concatenate([[0.0], np.cumsum(rest)])[:, None] * d[None, :]


# ------------------------------------------------------------------ solver

def _perp(d: np.ndarray) -> np.ndarray:
    return np.array([-d[1], d[0]])


def segment_angles(p: np.ndarray) -> np.ndarray:
    """Signed turn at each interior node (rad), + = counter-clockwise."""
    d = p[1:] - p[:-1]
    d1, d2 = d[:-1], d[1:]
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    dot = (d1 * d2).sum(axis=1)
    return np.arctan2(cross, dot)


class ArmChain:
    """State: positions p (N,2), velocities v (N,2); the bend multipliers
    are reset every substep (one pass each), as in Small Steps."""

    def __init__(self, geom: ArmGeometry, params: ChainParams, p0: np.ndarray,
                 pinned: tuple[int, ...] = (0, 1)):
        self.geom = geom
        self.prm = params
        self.p = p0.astype(np.float64).copy()
        self.v = np.zeros_like(self.p)
        self.m = geom.masses()
        self.w = 1.0 / self.m
        self.w[list(pinned)] = 0.0
        self.rest = geom.rest_lengths()
        self.radii = geom.radii()
        self.s_node = np.concatenate([[0.0], np.cumsum(self.rest)])
        self.theta_target = np.zeros(N_SEG - 1)
        r_ratio = self.radii[1:-1] / geom.r_base_m
        self.k_active = params.k_bend * r_ratio ** 3
        self.k_passive = params.k_passive_frac * params.k_bend * r_ratio ** 3
        self.activation = np.zeros(N_SEG - 1)
        # drag areas per node: half of each adjacent segment
        l_node = np.zeros(N_NODES)
        l_node[:-1] += 0.5 * self.rest
        l_node[1:] += 0.5 * self.rest
        self.K_per = 0.5 * geom.rho_water * (2.0 * self.radii * l_node) * params.c_per
        self.K_tan = 0.5 * geom.rho_water * (2.0 * np.pi * self.radii * l_node) * params.c_tan
        self.time = 0.0
        self.sub_count = 0

    # -- activation --------------------------------------------------------
    def set_activation_wave(self, cmd: ReachCommand, t: float) -> None:
        """Yekutieli's sigmoid front at the interior nodes' arc lengths."""
        L = self.geom.length_m
        V = L / cmd.t_act
        wdt = cmd.front_width_m if cmd.front_width_m is not None else L / N_SEG
        s_front = V * (t - cmd.t_start)
        a = 0.5 * (1.0 + np.tanh((s_front - self.s_node[1:-1]) / wdt))
        if not cmd.hold:
            # activation fades over 0.4 L behind the front (Yekutieli I's early termination)
            a = a * 0.5 * (1.0 + np.tanh((self.s_node[1:-1] - (s_front - 0.4 * L)) / wdt))
        self.activation = cmd.a_ca * a

    # -- one substep ----------------------------------------------------------
    def substep(self, dt: float) -> None:
        prm = self.prm
        p, v, w = self.p, self.v, self.w
        # external: gravity minus buoyancy
        if prm.g_eff:
            v[:, 1] -= dt * prm.g_eff * (w > 0)
        # drag, per node on the local tangent, implicit per component
        tang = np.empty_like(p)
        tang[1:-1] = p[2:] - p[:-2]
        tang[0] = p[1] - p[0]
        tang[-1] = p[-1] - p[-2]
        tang /= np.linalg.norm(tang, axis=1)[:, None]
        norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        v_t = (v * tang).sum(axis=1)
        v_n = (v * norm).sum(axis=1)
        f_t = 1.0 / (1.0 + dt * self.K_tan * np.abs(v_t) * self.w_or_one())
        f_n = 1.0 / (1.0 + dt * self.K_per * np.abs(v_n) * self.w_or_one())
        v_t *= f_t
        v_n *= f_n
        v[:] = v_t[:, None] * tang + v_n[:, None] * norm
        v[w == 0] = 0.0
        # predict
        p_prev = p.copy()
        p += dt * v
        # constraints, Gauss-Seidel passes, base -> tip
        for _ in range(self.prm.n_pass):
            self._project_distance(dt)
            self._project_bend(dt, p_prev)
        # velocity update
        v[:] = (p - p_prev) / dt
        v[w == 0] = 0.0
        self.time += dt
        self.sub_count += 1

    def w_or_one(self) -> np.ndarray:
        return np.where(self.w > 0, self.w, 0.0)

    def _project_distance(self, dt: float) -> None:
        p, w = self.p, self.w
        alpha = (1.0 / self.prm.k_dist) / (dt * dt)
        for i in range(N_SEG):
            d = p[i + 1] - p[i]
            ln = np.sqrt(d @ d)
            if ln < 1e-12:
                continue
            C = ln - self.rest[i]
            wsum = w[i] + w[i + 1]
            if wsum == 0.0:
                continue
            dlam = -C / (wsum + alpha)
            g = d / ln
            p[i] -= w[i] * dlam * g
            p[i + 1] += w[i + 1] * dlam * g

    def _project_bend(self, dt: float, p_prev: np.ndarray) -> None:
        p, w = self.p, self.w
        gamma = self.prm.bend_damp_s / dt         # XPBD: gamma = alpha~ beta~ / dt with beta = bend_damp_s * k
        for j in range(N_SEG - 1):
            i0, i1, i2 = j, j + 1, j + 2
            d1 = p[i1] - p[i0]
            d2 = p[i2] - p[i1]
            l1 = d1 @ d1
            l2 = d2 @ d2
            if l1 < 1e-16 or l2 < 1e-16:
                continue
            theta = np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2)
            C = theta - self.theta_target[j]
            g0 = _perp(d1) / l1
            g2 = _perp(d2) / l2
            g1 = -g0 - g2
            k = self.k_passive[j] + self.activation[j] * self.k_active[j]
            alpha = (1.0 / k) / (dt * dt)
            denom = (1.0 + gamma) * (w[i0] * (g0 @ g0) + w[i1] * (g1 @ g1) + w[i2] * (g2 @ g2)) + alpha
            if denom == 0.0:
                continue
            damp = gamma * (g0 @ (p[i0] - p_prev[i0]) + g1 @ (p[i1] - p_prev[i1]) + g2 @ (p[i2] - p_prev[i2]))
            dlam = (-C - damp) / denom
            p[i0] += w[i0] * dlam * g0
            p[i1] += w[i1] * dlam * g1
            p[i2] += w[i2] * dlam * g2

    # -- a whole tick ------------------------------------------------------
    def tick(self, cmd: ReachCommand | None = None) -> None:
        dt = self.prm.dt_tick / self.prm.n_sub
        for _ in range(self.prm.n_sub):
            if cmd is not None:
                self.set_activation_wave(cmd, self.time)
            self.substep(dt)


# -------------------------------------------------------------- measuring

@dataclass
class Trace:
    """What a run records, once per tick."""
    t: list[float] = field(default_factory=list)
    p: list[np.ndarray] = field(default_factory=list)
    activation: list[np.ndarray] = field(default_factory=list)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array(self.t), np.array(self.p), np.array(self.activation)


def run_reach(geom: ArmGeometry, prm: ChainParams, cmd: ReachCommand, p0: np.ndarray,
              duration_s: float, pinned: tuple[int, ...] = (0, 1)) -> Trace:
    arm = ArmChain(geom, prm, p0, pinned=pinned)
    tr = Trace()
    n_ticks = int(round(duration_s / prm.dt_tick))
    for _ in range(n_ticks + 1):
        tr.t.append(arm.time)
        tr.p.append(arm.p.copy())
        tr.activation.append(arm.activation.copy())
        arm.tick(cmd)
    return tr


def run_passive(geom: ArmGeometry, prm: ChainParams, p0: np.ndarray, v0: np.ndarray,
                duration_s: float, pinned: tuple[int, ...] = (0, 1)) -> Trace:
    """A dead arm: no activation, an initial velocity field (the whip)."""
    arm = ArmChain(geom, prm, p0, pinned=pinned)
    arm.v[:] = v0
    arm.v[arm.w == 0] = 0.0
    tr = Trace()
    n_ticks = int(round(duration_s / prm.dt_tick))
    for _ in range(n_ticks + 1):
        tr.t.append(arm.time)
        tr.p.append(arm.p.copy())
        tr.activation.append(arm.activation.copy())
        arm.tick(None)
    return tr


def bend_point(p: np.ndarray, s_node: np.ndarray) -> tuple[float, np.ndarray, float]:
    """The point of maximum curvature: (arc length, position, |turn|) --
    curvature interpolated parabolically between the three nodes around the
    max so the bend position moves smoothly, as a tracked marker would."""
    ang = np.abs(segment_angles(p))          # at interior nodes 1..N-2
    j = int(np.argmax(ang))
    k = j + 1                                 # node index
    if 0 < j < len(ang) - 1:
        y0, y1, y2 = ang[j - 1], ang[j], ang[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        off = float(np.clip(off, -1.0, 1.0))
    else:
        off = 0.0
    s = s_node[k] + off * (s_node[k + 1] - s_node[k] if off > 0 else s_node[k] - s_node[k - 1])
    pos = p[k] + off * ((p[k + 1] - p[k]) if off > 0 else (p[k] - p[k - 1]))
    return float(s), pos, float(ang[j])


BEND_MIN_RAD = np.radians(8.0)   # below this per-node turn there is no bend to track


def bend_kinematics(tr: Trace, s_node: np.ndarray, smooth_ticks: int = 3):
    """Per tick: bend arc length, bend position, bend tangential speed (the
    marker's spatial speed, what Sumbre 2001 measured), smoothed. Ticks with
    no bend (max turn < BEND_MIN_RAD) carry NaN: the bend has dissolved."""
    t, P, _ = tr.arrays()
    s = np.full(len(t), np.nan)
    pos = np.full((len(t), 2), np.nan)
    amp = np.empty(len(t))
    for i in range(len(t)):
        s_i, pos_i, amp[i] = bend_point(P[i], s_node)
        if amp[i] >= BEND_MIN_RAD:
            s[i], pos[i] = s_i, pos_i
    vel = np.full(len(t), np.nan)
    vel[1:-1] = np.linalg.norm(pos[2:] - pos[:-2], axis=1) / (t[2:] - t[:-2])
    if smooth_ticks > 1:
        k = np.ones(smooth_ticks) / smooth_ticks
        ok = np.isfinite(vel)
        v0 = np.where(ok, vel, 0.0)
        num = np.convolve(v0, k, mode="same")
        den = np.convolve(ok.astype(float), k, mode="same")
        vel = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        vel[~ok] = np.nan
    return t, s, pos, vel, amp


def mean_local_angle_deg(p: np.ndarray, upto_node: int | None = None) -> float:
    """Mean |turn| over the interior nodes (Yekutieli's straightness measure,
    <= 2 deg = straight); `upto_node` restricts to nodes 1..upto_node-1."""
    ang = np.abs(np.degrees(segment_angles(p)))
    if upto_node is not None:
        ang = ang[: max(upto_node - 1, 0)]
    return float(ang.mean()) if len(ang) else 0.0
