"""#217 finding 1 / #72 -- THE EOS CHECKERBOARD BENCH (measurement only).

The gas solver carries an unstable cell-to-cell (odd-even, "checkerboard")
pattern: neighbouring tiles drift apart in density while the stored pressure
stays flat. This harness reproduces it on ``levels/bench_two_room`` (two
sealed 12x12 rooms, one door tile, one crate) and measures it per tick.

Scenarios (``--case``):
  still   all ambient, no seed: the control, must stay perfectly still.
  cb      every other tile +/-``--amp`` (default 0.1 %) bulk air, T ambient.
  warm    the left room +40 degrees, its air thinned to equal pressure
          (#217's first comment).

Every case runs at any tile size (``--tile``, metres; the level's tilemap is
kept and only ``tile_size_m`` changes, as the issue's probes did with their
TILE variable) and with EOS dial overrides (``--eos adiabatic_index=1.0001``
switches step 6's flow work off: k_work = (gamma-1)*T_AMB).

Metrics per tick, over the accountable gas cells (``gmap._gas_energy_
accountable()``):
  gap_rms    rms of (P_stored - P_state)/P_state, P_stored = ``atmosphere``,
             P_state = C*N*T_abs (the air's own pressure)
  gap_corr   lag-1 neighbour correlation of that gap (over every
             horizontally / vertically adjacent accountable pair)
  cb(f)      the amplitude of the (-1)^(x+y) component of f: the projection
             sum(s*(f - room mean))/sum(s^2) with s = (-1)^(x+y); for f =
             f0 + a*s it is a. Reported for N (relative), u_x, u_y (m/s),
             P_stored (relative), T (game degrees) and gas_energy/N (= T_abs)
  T min/max  game degrees, relative to ambient, over the accountable cells
  |u| max    m/s

``--steps`` adds the per-step attribution (deliverable 2 of the #217 patch):
each tick is split, from the engine's own debug outputs and test-only
references, into the increments of step 1 (substeps: SL advection of u +
the donor-cell mass flux), step 3 (the solve: p* -> P), step 4 (the kick +
its absorb / sponge / cap / drag chain) and step 6+7 (energy), each
projected on the checkerboard mode. See ``attribute_tick``.

Seeding goes through the scenario-builder seam only: the bulk planes are
written BEFORE the first tick, then ``reseed_gas_energy`` /
``seed_gas_temperature`` restricted to the cells written (the gas energy
seam's documented builder route; a bare ``temperature[...] =`` goes vacuous,
#82).

HARNESS, not a pytest gate (``_`` prefix). The collected property test is
``tests/test_eos_checkerboard.py``.

Run:
    <breach-py> tests/_eos_checkerboard_bench.py --case cb --tile 0.333 --seconds 20
    <breach-py> tests/_eos_checkerboard_bench.py --table            # the issue's table
    <breach-py> tests/_eos_checkerboard_bench.py --case cb --steps --ticks 40
    <breach-py> tests/_eos_checkerboard_bench.py --solve-bias     # step 3 alone
    <breach-py> tests/_eos_checkerboard_bench.py --case warm --seconds 40
    <breach-py> tests/_eos_checkerboard_bench.py --stability smoke_light_studio --ticks 72
    <breach-py> tests/_eos_checkerboard_bench.py --stability smoke_light_studio --ticks 768 --tps 192 --every 16
    <breach-py> tests/_eos_checkerboard_bench.py --arena --seconds 30 --every 24
    <breach-py> tests/_eos_checkerboard_bench.py --stability fire_studio --detonate 24,16,24 --ticks 120

``--stability`` is the blast-regime trace (section 8 of the design doc): per
tick on a level's live path, the Courant numbers, N's extremes, the stored P
against the state pressure, the solve's linearized compression and the
kick loop's kinetic-energy budget. See ``stability_trace``.

Findings and tables: docs/eos_checkerboard_design_2026-11-20.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import temperature_scale  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402

FIXTURE = "bench_two_room"
Q = 65536.0
TPS = int(getattr(CFG.clock, "ticks_per_second", 24))


# ---------------------------------------------------------------------------
# scenario
# ---------------------------------------------------------------------------
def build(tile_m: float = 0.333, case: str = "cb", amp: float = 1e-3,
          warm_deg: float = 40.0, eos: dict | None = None, seed: int = 1):
    """Load the fixture at ``tile_m`` and seed ``case``. Returns the sim.

    Seeding happens before the first tick: the bulk planes are scaled
    directly and the stored energy is re-derived on exactly the cells
    written (``reseed_gas_energy`` / ``seed_gas_temperature``)."""
    level = load_level(FIXTURE, levels_dir=str(ROOT / "levels"))
    level = dataclasses.replace(level, tile_size_m=float(tile_m))
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    for k, v in (eos or {}).items():
        setattr(sim.physics_runner.eos, k, type(getattr(sim.physics_runner.eos, k))(v))
    g = sim.gmap
    acc = g._gas_energy_accountable()
    cons = np.flatnonzero(g.gases.conservative)
    if case == "still":
        pass
    elif case == "cb":
        yy, xx = np.indices(acc.shape)
        s = np.where((xx + yy) % 2 == 0, 1.0, -1.0)
        fac = 1.0 + amp * s
        for gi in cons:
            g.gas[gi][acc] = np.rint(g.gas[gi][acc].astype(np.float64)
                                     * fac[acc]).astype(g.gas.dtype)
        g.reseed_gas_energy(acc)
    elif case == "warm":
        left = acc & (np.indices(acc.shape)[1] < door_column(g))
        t_amb = temperature_scale.load(CFG).eos_t_amb_k
        thin = t_amb / (t_amb + warm_deg)
        for gi in cons:
            g.gas[gi][left] = np.rint(g.gas[gi][left].astype(np.float64)
                                      * thin).astype(g.gas.dtype)
        g.seed_gas_temperature(left, int(round(warm_deg * Q)))
    else:
        raise ValueError(f"unknown case {case!r}")
    return sim


def door_column(g) -> int:
    """The partition's column: the interior column that is solid on all rows
    but one (the door). Read from the map, never assumed."""
    sol = g.solid
    h, w = sol.shape
    best, best_n = w // 2, -1
    for x in range(1, w - 1):
        n = int(sol[1:h - 1, x].sum())
        if n > best_n:
            best, best_n = x, n
    return best


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class Probe:
    """The per-tick metrics, over the accountable cells."""

    def __init__(self, sim):
        self.sim = sim
        g = sim.gmap
        self.acc = g._gas_energy_accountable()
        h, w = self.acc.shape
        yy, xx = np.indices((h, w))
        self.s = np.where((xx + yy) % 2 == 0, 1.0, -1.0)
        self.t_amb = temperature_scale.load(CFG).eos_t_amb_k
        # rooms: connected components of the accountable set with the door
        # tile cut out, so a per-room mean can be removed before projecting
        # (a warm room's mean is not a checkerboard).
        dc = door_column(g)
        rooms = np.zeros((h, w), dtype=np.int32)
        rooms[self.acc & (xx < dc)] = 1
        rooms[self.acc & (xx > dc)] = 2
        self.rooms = rooms
        # neighbour pairs (both accountable), for the lag-1 correlation
        a = self.acc
        self.pairs_e = a[:, :-1] & a[:, 1:]
        self.pairs_s = a[:-1, :] & a[1:, :]

    # --- raw planes, in real units -----------------------------------------
    def planes(self):
        g = self.sim.gmap
        n = g._gas_bulk_n_raw().astype(np.float64) / Q          # N (1.0 = ambient)
        t = g.temperature.astype(np.float64) / Q                 # game deg rel.
        p = g.atmosphere.astype(np.float64) / Q                  # atm
        ux = g.wind_x.astype(np.float64) / Q
        uy = g.wind_y.astype(np.float64) / Q
        e = g.gas_energy.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            eabs = np.where(n > 0, e / (n * Q) / Q, 0.0)         # E/N = T_abs (K)
        return dict(N=n, T=t, P=p, ux=ux, uy=uy, Eabs=eabs)

    def cb(self, f, rel=False):
        """Amplitude of the (-1)^(x+y) component of f over the rooms
        (each room's own mean removed first)."""
        num = den = 0.0
        mean_all = 0.0
        for r in (1, 2):
            m = self.rooms == r
            if not m.any():
                continue
            fr = f[m]
            mu = fr.mean()
            num += float((self.s[m] * (fr - mu)).sum())
            den += float(m.sum())
            mean_all += mu * m.sum()
        amp = num / den if den else 0.0
        if rel:
            amp /= (mean_all / den)
        return amp

    def gap(self, pl=None):
        pl = pl or self.planes()
        p_state = pl["N"] * (pl["T"] + self.t_amb) / self.t_amb
        with np.errstate(divide="ignore", invalid="ignore"):
            gp = np.where(self.acc & (p_state > 0), (pl["P"] - p_state) / p_state, 0.0)
        return gp

    def corr1(self, f):
        a = np.concatenate([f[:, :-1][self.pairs_e], f[:-1, :][self.pairs_s]])
        b = np.concatenate([f[:, 1:][self.pairs_e], f[1:, :][self.pairs_s]])
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    def row(self):
        pl = self.planes()
        gp = self.gap(pl)
        acc = self.acc
        umag = np.hypot(pl["ux"], pl["uy"])
        return dict(
            gap_rms=float(np.sqrt((gp[acc] ** 2).mean())),
            gap_corr=self.corr1(gp),
            cbN=self.cb(pl["N"], rel=True),
            cbux=self.cb(pl["ux"]), cbuy=self.cb(pl["uy"]),
            cbP=self.cb(pl["P"], rel=True),
            cbT=self.cb(pl["T"]),
            cbE=self.cb(pl["Eabs"]),
            Tmin=float(pl["T"][acc].min()), Tmax=float(pl["T"][acc].max()),
            umax=float(umag[acc].max()),
            Pspread=float(pl["P"][acc].max() - pl["P"][acc].min()),
        )


def step(sim):
    sim.set_paused(False)
    sim.step()


# ---------------------------------------------------------------------------
# per-step attribution (deliverable 2)
# ---------------------------------------------------------------------------
# The tick is split from what the engine exposes, never by changing it:
#   u0, N0, P0, T0   the planes before the tick
#   u1               step 1a's SL advection of u, replayed EXACTLY by the
#                    test-only reference `bp.eos_sl_advect_ref` (validated:
#                    its face-form divergence must equal the solver's cached
#                    div_u bit for bit)
#   N1, p*, div_u    the solver's own solve-input caches (`dbg_mg_inputs`):
#                    N1 is the post-step-1 bulk N (nothing after step 1 moves N)
#   P                the materialized `atmosphere` after the tick
#   u2               the stored wind after the tick (kick + absorb / sponge /
#                    cap / drag; nothing outside eos.step writes wind here)
#   T2               the mirror after step 6 + 7
# The step-1 mass flux is re-derived from (u1, N0) with the engine's own
# integer face law (bulk_transport.cpp: v_face = (u_i + u_j) >> 1, upwind
# donor, coeff = perm * (dt_s/dx), the transport step in tiles per m/s since
# #217 patch U, one >>32 truncation per plane) and checked
# against N1, then split into its central part (v_face * face-mean N) and its
# upwind part.
#
# The checkerboard functional. With s = (-1)^(x+y), a face flux F (lo -> hi)
# changes sum(s*N) by F*(s_hi - s_lo) = -2*s_lo*F, so the checkerboard of N is
# fed ONLY by the "staggered" component of the face fluxes, sum_f s_lo*F_f.
# For the central part F ~ coeff * (u_lo + u_hi)/2 * N, that is
#     Phi(u) = sum_{open faces} s_lo * (u_lo + u_hi) / 2     (m/s)
# Phi is linear in u, so every step's velocity increment has its own share of
# the next tick's checkerboard mass drift: Phi(u1 - u0) is step 1a's,
# Phi(u2 - u1) step 4's, and step 4 is split further into the pressure kick
# re-derived in float from the ENGINE's stored P (its 1/N with the cell's own N
# and with the room-mean N: the difference is the N-checkerboard x smooth
# pressure-gradient coupling), the same kick from a float re-solve of step 3
# on the engine's own solve inputs (no Q16 rounding of P: the difference
# between the two is what the stored pressure's quantization drives), and the
# remainder (drag / cap / the integer kick's floors).

def _open_faces(g):
    sol = g.solid
    oe = (~sol[:, :-1]) & (~sol[:, 1:])
    os_ = (~sol[:-1, :]) & (~sol[1:, :])
    return oe, os_


def phi(u_x, u_y, s, oe, os_):
    """Staggered-checkerboard content of a velocity field (see above)."""
    fe = 0.5 * (u_x[:, :-1] + u_x[:, 1:]) * s[:, :-1]
    fs = 0.5 * (u_y[:-1, :] + u_y[1:, :]) * s[:-1, :]
    return float(fe[oe].sum() + fs[os_].sum())


def _q(v):
    return int(np.floor(v * Q + 0.5)) if v >= 0 else -int(np.floor(-v * Q + 0.5))


def transport_step_q(dt, n_sub, dx):
    """eos_transport_step_q (eos_solver.cpp, #217 patch U), transcribed: the
    float32 dt and dx the engine holds, dt_s = dt / n_sub and the dx floor in
    double, ONE quantize of dt_s / dx (tiles per m/s per substep)."""
    dx_d = max(float(np.float32(dx)), 1e-6)
    return _q(float(np.float32(dt)) / float(n_sub) / dx_d)


def mass_flux_faces(g, ux_raw, uy_raw, n_planes, dt, dx, n_sub=1):
    """The step-1d donor-cell face flux, per conservative plane, in raw Q16
    (bulk_transport.cpp:133-142 for n_sub == 1). Returns (dq_e, dq_s) summed
    over the planes, each split (total, central) as float raw counts."""
    sol = g.solid
    perm = g.dyn_permeability.astype(np.float64)
    dt_q = transport_step_q(dt, n_sub, dx)   # #217 patch U: dt_s/dx, tiles
    oe, os_ = _open_faces(g)
    pe = np.minimum(perm[:, :-1], perm[:, 1:])
    ps = np.minimum(perm[:-1, :], perm[1:, :])
    ce = np.where(oe & (pe > 0), (np.vectorize(_q)(pe).astype(np.int64) * dt_q) >> 16, 0)
    cs = np.where(os_ & (ps > 0), (np.vectorize(_q)(ps).astype(np.int64) * dt_q) >> 16, 0)
    ux = ux_raw.astype(np.int64)
    uy = uy_raw.astype(np.int64)
    ve = (ux[:, :-1] + ux[:, 1:]) >> 1
    vs = (uy[:-1, :] + uy[1:, :]) >> 1
    tot_e = np.zeros(ve.shape, np.int64)
    tot_s = np.zeros(vs.shape, np.int64)
    cen_e = np.zeros(ve.shape, np.float64)
    cen_s = np.zeros(vs.shape, np.float64)
    for N in n_planes:
        N = N.astype(np.int64)
        de = np.where(ve > 0, N[:, :-1], N[:, 1:])
        ds = np.where(vs > 0, N[:-1, :], N[1:, :])
        tot_e += (ve * de * ce) >> 32
        tot_s += (vs * ds * cs) >> 32
        cen_e += ve * 0.5 * (N[:, :-1] + N[:, 1:]) * ce / 2.0 ** 32
        cen_s += vs * 0.5 * (N[:-1, :] + N[1:, :]) * cs / 2.0 ** 32
    return (tot_e, tot_s), (cen_e, cen_s)


def apply_div(shape, fe, fs):
    """N change of a face-flux set (lo loses, hi gains)."""
    d = np.zeros(shape, np.float64)
    d[:, :-1] -= fe
    d[:, 1:] += fe
    d[:-1, :] -= fs
    d[1:, :] += fs
    return d


def grad_wide(P, sol):
    """Step 4's gradient stencil: (P[ir] - P[il]) / (2 dx) with the solid
    mirror (mirror_idx returns SELF at a wall), eos_solver.cpp:1059-1064.
    Returned without the 1/(2dx) factor."""
    h, w = P.shape
    Pp = np.pad(P, 1, mode="edge")
    sp = np.pad(sol, 1, constant_values=True)
    c = P
    r = np.where(sp[1:-1, 2:], c, Pp[1:-1, 2:])
    l = np.where(sp[1:-1, :-2], c, Pp[1:-1, :-2])
    d = np.where(sp[2:, 1:-1], c, Pp[2:, 1:-1])
    u = np.where(sp[:-2, 1:-1], c, Pp[:-2, 1:-1])
    return r - l, d - u


def float_solve(g, eos, pstar, div_u, n1, dt):
    """Step 3 in float64 (space map: no ring, no sponge): the level-0 SPD
    operator of mg_build_levels, eos_solver.cpp:1821-1882, solved exactly.
        m_i P_i + sum_f g_f (P_i - P_j) = m_i rhs_i
        m = 1/aK, aK = gamma p* K dt^2/dx^2, rhs = p* - gamma p* dt div u,
        g_f = perm_f / ((N_i + N_j)/2)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    sol = g.solid
    h, w = sol.shape
    gam = float(eos.adiabatic_index)
    K = float(eos.c_max) ** 2 / gam
    dx = float(eos.dx)
    perm = g.dyn_permeability.astype(np.float64)
    idx = -np.ones((h, w), np.int64)
    cells = np.flatnonzero(~sol.ravel())
    idx.ravel()[cells] = np.arange(cells.size)
    aK = np.maximum(gam * pstar * K * dt * dt / (dx * dx), 1e-30)
    m = 1.0 / aK
    rhs = pstar - gam * pstar * dt * div_u
    rows, cols, vals = [], [], []
    diag = m.copy()
    for (sl_a, sl_b, pf) in (
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None)),
             np.minimum(perm[:, :-1], perm[:, 1:])),
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None)),
             np.minimum(perm[:-1, :], perm[1:, :]))):
        ia, ib = idx[sl_a], idx[sl_b]
        ok = (ia >= 0) & (ib >= 0) & (pf > 0)
        nh = np.maximum(0.5 * (n1[sl_a] + n1[sl_b]), float(eos.N_FLOOR_SOLVER))
        gf = np.where(ok, pf / nh, 0.0)[ok]
        a, b = ia[ok], ib[ok]
        rows += [a, b, a, b]
        cols += [b, a, a, b]
        vals += [-gf, -gf, gf, gf]
    n = cells.size
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    A = A + sp.diags(diag.ravel()[cells])
    x = spl.spsolve(A.tocsc(), (m * rhs).ravel()[cells])
    P = np.zeros((h, w))
    P.ravel()[cells] = x
    return P


def attribute_tick(sim, pr):
    """Run ONE tick and split it per step (see the block comment above).
    Returns a dict of checkerboard projections (relative N amplitude units
    for the mass terms, m/s for Phi, degrees for T)."""
    g = sim.gmap
    eos = sim.physics_runner.eos
    dt = 1.0 / TPS
    s = pr.s
    oe, os_ = _open_faces(g)
    cons = np.flatnonzero(g.gases.conservative)
    # --- before ---
    u0x, u0y = g.wind_x.copy(), g.wind_y.copy()
    T0 = g.temperature.copy()
    n_planes0 = [g.gas[gi].copy() for gi in cons]
    N0 = sum(p.astype(np.int64) for p in n_planes0)
    pl0 = pr.planes()
    # --- the tick ---
    step(sim)
    n_sub = int(eos.dbg_last_n_sub)
    h, w = g.solid.shape
    pstar, div_u, n_tot = (a.reshape(h, w) for a in eos.dbg_mg_inputs())
    N1 = n_tot.astype(np.int64)
    pl2 = pr.planes()
    # --- step 1a replay: u1 ---
    u1x, u1y, t_scr = u0x.copy(), u0y.copy(), T0.copy()
    bp.eos_sl_advect_ref(u1x, u1y, t_scr, g.solid, g.is_vacuum,
                         g.dyn_permeability, float(np.float32(dt)), n_sub,
                         dx=float(eos.dx), thermal_solid=g.thermal_solid)
    # validation 1: the face-form divergence of u1 == the solver's div_u
    sol = g.solid
    def f2(u, axis):
        up = np.pad(u.astype(np.int64), 1)
        spd = np.pad(sol, 1, constant_values=True)
        if axis == 1:
            fe = np.where(spd[1:-1, 2:], 0, up[1:-1, 1:-1] + up[1:-1, 2:])
            fw = np.where(spd[1:-1, :-2], 0, up[1:-1, 1:-1] + up[1:-1, :-2])
        else:
            fe = np.where(spd[2:, 1:-1], 0, up[1:-1, 1:-1] + up[2:, 1:-1])
            fw = np.where(spd[:-2, 1:-1], 0, up[1:-1, 1:-1] + up[:-2, 1:-1])
        return fe - fw
    inv2dx = _q(1.0 / (2.0 * float(eos.dx)))
    dv = ((f2(u1x, 1).astype(np.int64) * inv2dx) >> 16) + ((f2(u1y, 0).astype(np.int64) * inv2dx) >> 16)
    live = ~(sol | g.is_vacuum)
    div_ok = bool(np.array_equal(dv[live], div_u[live].astype(np.int64)))
    # --- step 1d: the mass flux, re-derived and checked ---
    mass_ok = None
    if n_sub == 1:
        (te, ts), (ce_, cs_) = mass_flux_faces(g, u1x, u1y, n_planes0, dt,
                                               float(eos.dx), n_sub)
        dN = apply_div(N0.shape, te, ts)
        mass_ok = bool(np.array_equal((N0 + dN.astype(np.int64))[live], N1[live]))
        dN_cen = apply_div(N0.shape, ce_, cs_)
        dN_up = dN - dN_cen
    else:
        dN = (N1 - N0).astype(np.float64)
        dN_cen = dN_up = np.zeros_like(dN)
    mean_n = float(N0[pr.acc].mean())
    d_cbN = pr.cb((N1 - N0).astype(np.float64)) / mean_n
    d_cbN_cen = pr.cb(dN_cen) / mean_n
    d_cbN_up = pr.cb(dN_up) / mean_n
    # --- velocities in m/s ---
    U0 = (u0x / Q, u0y / Q)
    U1 = (u1x / Q, u1y / Q)
    U2 = (g.wind_x / Q, g.wind_y / Q)
    phi0 = phi(*U0, s, oe, os_)
    phi1 = phi(*U1, s, oe, os_)
    phi2 = phi(*U2, s, oe, os_)
    # --- step 4 split ---
    gam = float(eos.adiabatic_index)
    K = float(eos.c_max) ** 2 / gam
    dx = float(eos.dx)
    n1f = N1 / Q
    n1_room = np.zeros_like(n1f)
    for r in (1, 2):
        m = pr.rooms == r
        if m.any():
            n1_room[m] = n1f[m].mean()
    n1_room[n1_room == 0] = 1.0
    P_eng = g.atmosphere / Q
    gx, gy = grad_wide(P_eng, sol)
    kick = lambda gx_, gy_, nn: (-dt * K * gx_ / (2 * dx) / nn, -dt * K * gy_ / (2 * dx) / nn)
    kx, ky = kick(gx, gy, np.maximum(n1f, 1e-3))
    kxm, kym = kick(gx, gy, n1_room)
    P_f = float_solve(g, eos, pstar / Q, div_u / Q, n1f, dt)
    gfx, gfy = grad_wide(P_f, sol)
    kfx, kfy = kick(gfx, gfy, np.maximum(n1f, 1e-3))
    for a in (kx, ky, kxm, kym, kfx, kfy):
        a[sol] = 0.0
    phi_kick = phi(kx, ky, s, oe, os_)
    phi_kick_meanN = phi(kxm, kym, s, oe, os_)
    phi_kick_float = phi(kfx, kfy, s, oe, os_)
    phi_rest = (phi2 - phi1) - phi_kick
    # P quantization: the stored P minus the float solve, in counts
    dq = (P_eng - P_f)[pr.acc] * Q
    # --- T: step 1 vs steps 6+7 ---
    t_amb = pr.t_amb
    with np.errstate(divide="ignore", invalid="ignore"):
        T1 = np.where(N1 > 0, (pstar / Q) * t_amb / (N1 / Q) - t_amb, 0.0)
    return dict(
        n_sub=n_sub, div_ok=div_ok, mass_ok=mass_ok,
        cbN0=pr.cb(pl0["N"], rel=True), cbN1=pr.cb(N1 / Q, rel=True),
        d_cbN=d_cbN, d_cbN_cen=d_cbN_cen, d_cbN_up=d_cbN_up,
        phi0=phi0, d_phi_sl=phi1 - phi0, d_phi_step4=phi2 - phi1,
        phi_kick=phi_kick, phi_kick_meanN=phi_kick_meanN,
        phi_kick_float=phi_kick_float, phi_rest=phi_rest,
        cb_div=pr.cb(div_u / Q), cb_pstar=pr.cb(pstar / Q, rel=True),
        cbP_float=pr.cb(P_f, rel=True), cbP=pr.cb(P_eng, rel=True),
        Pq_rms_counts=float(np.sqrt((dq ** 2).mean())),
        Pf_spread_counts=float((P_f[pr.acc].max() - P_f[pr.acc].min()) * Q),
        cbT0=pr.cb(pl0["T"]), cbT1=pr.cb(np.where(pr.acc, T1, 0.0)),
        cbT2=pr.cb(pl2["T"]),
    )


def kick_phi_kernel(g, s):
    """W with Phi(kick(P)) = -sum(W * P) * (dt K / (2 dx N)): the staggered
    face-flux content step 4's wide-stencil kick derives from a unit pressure
    at each cell (N = 1). Interior entries are exactly 0 (the wide gradient
    is blind to the odd-even mode); only the two cell layers along a wall or
    a door carry weight, alternating along the wall."""
    sol = g.solid
    oe, os_ = _open_faces(g)
    h, w = sol.shape
    W = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if sol[i, j]:
                continue
            P = np.zeros((h, w))
            P[i, j] = 1.0
            gx, gy = grad_wide(P, sol)
            gx[sol] = 0.0
            gy[sol] = 0.0
            W[i, j] = phi(gx, gy, s, oe, os_)
    return W


def solve_bias(tiles=(1.0, 0.5, 0.333, 0.25),
               amps=(0.08, 0.04, 0.02, 0.01, 0.001, -0.001, -0.01, -0.02, -0.04, -0.08)):
    """Step 3 in isolation (the test-only `bp.eos_mg_solve_ref`, the SAME
    mg_build_levels + mg_run_solve_cpu the live step runs) against its float
    twin (`float_solve`), on a synthetic still state: p* = N = 1 + a*s,
    T ambient, div u = 0, warm start at ambient. Reported: the restoring
    signal the kick can see, sum(W * P) over the wall layers, in Q16 counts of
    P. The float solve's signal follows the sign of a (restoring); the
    integer solve's does not."""
    print("# step 3 in isolation: sum(W*P) in Q16 counts (W = kick_phi_kernel);"
          " the float solve is the exact solution of the same operator")
    print(f"{'tile':>6} {'sigma^2':>8} " + " ".join(f"{a:+8.3f}" for a in amps))
    for tile in tiles:
        sim = build(tile, "still", 0.0, eos={"adiabatic_index": 1.0001})
        step(sim)   # binds eos.dx
        g = sim.gmap
        eos = sim.physics_runner.eos
        pr = Probe(sim)
        W = kick_phi_kernel(g, pr.s)
        live = ~g.solid
        h, w = live.shape
        sig2 = (float(eos.c_max) / TPS / float(eos.dx)) ** 2
        ints, flts = [], []
        for a in amps:
            ps = np.where(live, np.rint(Q * (1.0 + a * pr.s)), 0).astype(np.int32)
            dv = np.zeros((h, w), np.int32)
            pp = np.where(live, int(Q), 0).astype(np.int32)
            out = np.zeros((h, w), np.int32)
            bp.eos_mg_solve_ref(eos, ps, dv, ps.copy(), pp, g.solid, g.is_vacuum,
                                g.dyn_permeability.astype(np.float32),
                                np.float32(1.0 / TPS), out)
            Pf = float_solve(g, eos, ps / Q, dv / Q, ps / Q, 1.0 / TPS)
            ints.append(float((W * out).sum()))
            flts.append(float((W * Pf).sum() * Q))
        print(f"{tile:6.3f} {sig2:8.0f} int   " + " ".join(f"{v:+8.2f}" for v in ints))
        print(f"{'':6} {'':8} float " + " ".join(f"{v:+8.2f}" for v in flts))
        interior = live & (W == 0)
        print(f"{'':6} {'':8} (kernel: {int((W != 0).sum())} weighted cells, "
              f"{int(interior.sum())} interior cells at exactly 0)")


def run_steps(tile_m=0.333, case="cb", amp=1e-3, ticks=40, eos=None, every=1):
    sim = build(tile_m, case, amp, eos=eos)
    pr = Probe(sim)
    print(f"# per-step attribution: case={case} tile={tile_m} amp={amp} eos={eos or {}}")
    print("  dcbN = step 1's change of the relative N checkerboard (%), split central/upwind;"
          " Phi in m/s (see block comment); P counts = Q16 LSBs")
    hdr = (f"{'tick':>4} {'ok':>3} {'cbN%':>8} {'dcbN%':>8} {'cen%':>8} {'up%':>8} "
           f"{'Phi(u0)':>8} {'dPhiSL':>7} {'dPhi4':>8} {'kick':>8} {'kick_mN':>8} "
           f"{'kick_fP':>8} {'rest':>8} {'cbDiv':>8} {'cbp*%':>8} {'cbPf':>9} "
           f"{'Pq_rms':>6} {'Pf_spr':>6} {'cbT0':>7} {'cbT1':>7} {'cbT2':>7}")
    print(hdr)
    rows = []
    for t in range(1, ticks + 1):
        r = attribute_tick(sim, pr)
        rows.append(r)
        if t % every == 0 or t == ticks:
            ok = ("y" if r["div_ok"] else "n") + ("y" if r["mass_ok"] else
                                                   ("-" if r["mass_ok"] is None else "n"))
            print(f"{t:4d} {ok:>3} {100*r['cbN0']:8.4f} {100*r['d_cbN']:8.4f} "
                  f"{100*r['d_cbN_cen']:8.4f} {100*r['d_cbN_up']:8.4f} "
                  f"{r['phi0']:8.3f} {r['d_phi_sl']:7.3f} {r['d_phi_step4']:8.3f} "
                  f"{r['phi_kick']:8.3f} {r['phi_kick_meanN']:8.3f} "
                  f"{r['phi_kick_float']:8.4f} {r['phi_rest']:8.3f} "
                  f"{r['cb_div']:8.4f} {100*r['cb_pstar']:8.4f} {r['cbP_float']*Q:9.4f} "
                  f"{r['Pq_rms_counts']:6.2f} {r['Pf_spread_counts']:6.2f} "
                  f"{r['cbT0']:7.3f} {r['cbT1']:7.3f} {r['cbT2']:7.3f}")
    return rows


def run(tile_m=0.333, case="cb", amp=1e-3, ticks=480, eos=None, every=1,
        quiet=False):
    sim = build(tile_m, case, amp, eos=eos)
    pr = Probe(sim)
    rows = [dict(tick=0, **pr.row())]
    for t in range(1, ticks + 1):
        step(sim)
        if t % every == 0 or t == ticks:
            rows.append(dict(tick=t, **pr.row()))
    if not quiet:
        print_rows(rows, f"case={case} tile={tile_m} amp={amp} eos={eos or {}}")
    return rows, sim


def print_rows(rows, title):
    print(f"# {title}")
    print(f"{'tick':>5} {'gap_rms%':>9} {'corr':>6} {'cbN%':>9} {'cbP%':>9} "
          f"{'cbux':>8} {'cbuy':>8} {'cbT':>8} {'Tmin':>7} {'Tmax':>7} {'|u|max':>7}")
    for r in rows:
        print(f"{r['tick']:5d} {100*r['gap_rms']:9.4f} {r['gap_corr']:6.2f} "
              f"{100*r['cbN']:9.4f} {100*r['cbP']:9.5f} {r['cbux']:8.4f} "
              f"{r['cbuy']:8.4f} {r['cbT']:8.3f} {r['Tmin']:7.2f} {r['Tmax']:7.2f} "
              f"{r['umax']:7.3f}")


def table(case="cb", amp=1e-3, ticks=480, eos=None):
    """One summary row per setting, at the last tick (the issue's table)."""
    eos = eos or {}
    legs = [(t, eos) for t in (1.0, 0.5, 0.333, 0.25)]
    legs.append((0.333, {**eos, "adiabatic_index": 1.0001}))
    print(f"# case={case} amp={amp} ticks={ticks} ({ticks / TPS:.0f} s)")
    print(f"{'tile':>6} {'eos':<28} {'gap_rms%':>9} {'corr':>6} {'cbN%':>8} "
          f"{'cbT':>7} {'Tmin':>7} {'Tmax':>7} {'|u|max':>7}")
    for tile, e in legs:
        rows, _ = run(tile, case, amp, ticks, eos=e, every=ticks, quiet=True)
        r = rows[-1]
        print(f"{tile:6.3f} {str(e or 'shipped'):<28} {100 * r['gap_rms']:9.3f} "
              f"{r['gap_corr']:6.2f} {100 * r['cbN']:8.3f} {r['cbT']:7.2f} "
              f"{r['Tmin']:7.2f} {r['Tmax']:7.2f} {r['umax']:7.3f}")


# ---------------------------------------------------------------------------
# --stability: the blast-regime trace (#217 section 8, phase 1)
# ---------------------------------------------------------------------------
def _ke_books(n_raw, wx, wy, k_ke):
    """Sum N*k_ke*|u|^2 in the gas_energy currency (N_raw * dT_Q16): the
    kick loop's own ke_energy() bracket, in float, over the cells given."""
    u2 = (wx.astype(np.float64) / Q) ** 2 + (wy.astype(np.float64) / Q) ** 2
    return float((n_raw.astype(np.float64) * k_ke * u2 * Q).sum())


def stability_trace(level, ticks, tile=None, tps=None, eos=None, arena=False,
                    detonate=None, every=1, npz=None):
    """Per tick on the LIVE path of a level: the substep count, the tick's
    Courant numbers, how far N leaves ambient, the mass-weighted temperature,
    the stored P against the state pressure, the solve's linearized
    compression and the kinetic-energy budget of the kick loop.

    Columns (fractions of the accountable set's tick-0 thermal energy E0, %):
      Csub / Ctick  max|u_entry| * dt_s / dx and * dt / dx (a substep's SL
                    and flux displacement is bounded by max|u_entry|: the SL
                    sample is a convex combination, the face mean too)
      eps           gamma * dt * max|div u*| over cells with N >= 5 % N0:
                    the solve's rhs is p*(1 - eps); above 1 it goes negative
      kick/drag/clamp/absorb  e_kick_ke_sum, e_drag_heat_sum,
                    e_clamp_destroyed_sum, e_absorb_export_sum
      dKEu          the KE change NO counter books (step 1, other slots)
      k / Cupd / epsk  (#217 §8) the tick's sub-tick count, the Courant
                    number per pressure update max|u_entry| * dt / (k dx) and
                    the per-solve compression eps / k (k = 1 on a pre-§8
                    build); Csub is per substep of dt / (k n_sub);
      caph          (#217 §8.6.5) e_cap_heat_sum, the cap's returned heat
    ``tps`` (emulation only) runs the whole game at another tick rate, so
    the EOS's one pressure update per tick happens tps/24 times as often;
    the marine's turn law refuses high rates, and no unit turns in these
    scenes, so its check falls back to 0 there. ``arena`` builds the
    sealed-box scene of tests/_sealedbox_bisect_bench.py (test_arena, no
    vents, one glass ring, no fire). ``detonate`` = (x, y, tick at 24 tps):
    a frag through the payload executor, as tools/wind_drift_probe.py."""
    if tps:
        CFG.clock.ticks_per_second = tps
        import simulation.turning as _turning
        _orig = _turning.turn_accel_q16

        def _turn_accel(cfg):
            try:
                return _orig(cfg)
            except ValueError:
                return 0
        _turning.turn_accel_q16 = _turn_accel
    tps = float(CFG.clock.ticks_per_second)
    if arena:
        lvl = load_level("test_arena")
        lvl = dataclasses.replace(lvl, entities=[
            e for e in lvl.entities if e.class_name not in ("vent", "duct")])
    else:
        lvl = load_level(level)
    if tile is not None:
        lvl = dataclasses.replace(lvl, tile_size_m=tile)
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    if arena:
        from _sealedbox_bisect_bench import AQ_BOX
        from simulation import materials
        r0, r1, c0, c1 = AQ_BOX
        sim.gmap.seal_tiles([(r, c) for r in range(r0, r1 + 1)
                             for c in range(c0, c1 + 1)
                             if min(r - r0, r1 - r, c - c0, c1 - c) == 0],
                            materials.MAT_GLASS)
    sim.set_paused(False)
    e = sim.physics_runner.eos
    for k, v in (eos or {}).items():
        setattr(e, k, type(getattr(e, k))(v))
    g = sim.gmap
    dx, dt = float(g.tile_size_m), 1.0 / tps
    gam = float(e.adiabatic_index)
    k_ke = gam * (gam - 1.0) * float(e.T_AMB_K) / (2.0 * float(e.c_max) ** 2)
    t_amb = int(g._gas_energy_t_amb_raw())
    acct = g._gas_energy_accountable()
    N0 = float(np.median(g._gas_bulk_n_raw()[acct]))
    E0 = float(g.gas_energy[acct].astype(np.float64).sum())
    print(f"# {('arena' if arena else level)} dx {dx} tps {tps:g} cells "
          f"{int(acct.sum())} N_SUB_MAX {e.N_SUB_MAX} k_drag2 {e.k_drag2:.4g}")
    print("tick nsub umax Csub Ctick Nmax Nmin Tmw Pmean p*mean eps "
          "KE% kick% drag% clamp% rail% dKEu% clamps k Cupd epsk caph%")
    rows, clamps0 = [], int(e.u_clamp_hits)
    for t in range(ticks):
        if detonate and t == int(round(detonate[2] * tps / 24.0)):
            from simulation.payloads import execute_payload
            pay = sim.weapons_tables.payload_for_ammo("grenade_frag")
            execute_payload(g, sim.edit_queue, sim.units, int(detonate[1]),
                            int(detonate[0]), pay, sim.draws, events=[],
                            kind="grenade")
        acct = g._gas_energy_accountable()
        ke0 = _ke_books(g._gas_bulk_n_raw()[acct], g.wind_x[acct],
                        g.wind_y[acct], k_ke)
        umax = float(np.sqrt(((g.wind_x.astype(np.float64) / Q) ** 2
                              + (g.wind_y.astype(np.float64) / Q) ** 2).max()))
        if sim.is_paused():
            sim.set_paused(False)
        sim.step()
        acct = g._gas_energy_accountable()
        n = g._gas_bulk_n_raw()
        na = n[acct].astype(np.float64)
        E = g.gas_energy[acct].astype(np.float64)
        ke1 = _ke_books(n[acct], g.wind_x[acct], g.wind_y[acct], k_ke)
        kick, drag = float(e.e_kick_ke_sum), float(e.e_drag_heat_sum)
        # #217 §8.6.5: the cap's whole removal is destroyed + returned heat
        clamp = float(e.e_clamp_destroyed_sum) + float(e.e_cap_heat_sum)
        booked = (kick - float(e.e_absorb_export_sum)
                  - float(e.e_sponge_export_sum) - clamp - drag)
        ps, dv, nt = e.dbg_mg_inputs()
        am = acct.ravel() & (nt >= 0.05 * N0)
        eps = float(np.abs(gam * dt * dv[am].astype(np.float64) / Q).max()) \
            if am.any() else 0.0
        nsub = int(e.dbg_last_n_sub)
        kt = int(getattr(e, "dbg_last_k", 1) or 1)          # #217 §8
        caph = float(getattr(e, "e_cap_heat_sum", 0))
        clamps = int(e.u_clamp_hits) - clamps0
        clamps0 = int(e.u_clamp_hits)
        r = dict(tick=t, nsub=nsub, umax=umax, csub=umax * dt / (kt * nsub * dx),
                 ctick=umax * dt / dx, nmax=na.max() / N0, nmin=na.min() / N0,
                 tmw=(E.sum() / max(na.sum(), 1.0) - t_amb) / Q,
                 pmean=float(g.atmosphere[acct].astype(np.float64).mean()) / Q,
                 pstar=E.sum() / Q / Q / float(e.T_AMB_K) / len(E), eps=eps,
                 ke=100 * ke1 / E0, kick=100 * kick / E0, drag=100 * drag / E0,
                 clamp=100 * clamp / E0, rail=100 * float(e.e_rail_sum) / E0,
                 dkeu=100 * ((ke1 - ke0) - booked) / E0, clamps=clamps,
                 k=kt, cupd=umax * dt / (kt * dx), epsk=eps / kt,
                 caph=100 * caph / E0)
        rows.append(r)
        if t % every == 0 or t == ticks - 1:
            print(" ".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k])
                           for k in r))
    if npz:
        np.savez(npz, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    return rows


def parse_eos(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="cb", choices=("still", "cb", "warm"))
    ap.add_argument("--tile", type=float, default=0.333)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("--eos", action="append", default=[],
                    help="EOS dial override key=value (repeatable)")
    ap.add_argument("--steps", action="store_true",
                    help="per-step attribution (one row per tick)")
    ap.add_argument("--solve-bias", action="store_true",
                    help="step 3 in isolation, integer vs float, per tile size")
    ap.add_argument("--table", action="store_true",
                    help="the issue's table: the --case seed at 1.0/0.5/0.333/0.25 m "
                         "plus the adiabatic_index=1.0001 control at 0.333 m")
    ap.add_argument("--stability", default=None, metavar="LEVEL",
                    help="the blast-regime trace on a level's live path "
                         "(section 8); with --tps, --arena, --detonate, --npz")
    ap.add_argument("--tps", type=float, default=None,
                    help="with --stability: emulate another tick rate")
    ap.add_argument("--arena", action="store_true",
                    help="with --stability: the sealed-box test_arena scene")
    ap.add_argument("--detonate", default=None, metavar="X,Y,TICK",
                    help="with --stability: a frag at (x, y) on that 24-tps tick")
    ap.add_argument("--level-tile", type=float, default=None,
                    help="with --stability: override the level's tile_size_m")
    ap.add_argument("--npz", default=None, help="with --stability: save rows")
    a = ap.parse_args(argv)
    if a.stability or a.arena:
        tps = a.tps or TPS
        ticks = a.ticks if a.ticks is not None else int(round((a.seconds or 3.0) * tps))
        det = tuple(float(v) for v in a.detonate.split(",")) if a.detonate else None
        stability_trace(a.stability, ticks, tile=a.level_tile, tps=a.tps,
                        eos=parse_eos(a.eos), arena=a.arena, detonate=det,
                        every=a.every if a.ticks or a.seconds else 1, npz=a.npz)
        return
    ticks = a.ticks if a.ticks is not None else int(round((a.seconds or 20.0) * TPS))
    if a.table:
        table(a.case, a.amp, ticks, parse_eos(a.eos))
        return
    if a.solve_bias:
        solve_bias()
        return
    if a.steps:
        run_steps(a.tile, a.case, a.amp, ticks, eos=parse_eos(a.eos), every=a.every)
    else:
        run(a.tile, a.case, a.amp, ticks, eos=parse_eos(a.eos), every=a.every)


if __name__ == "__main__":
    main()
