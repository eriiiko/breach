"""Periodic-box smoke solver + the looping detail-volume bake (issue #90).

THE smoke-volume baker (design: docs/volumetric_smoke_design_2026-10-14.md
§2.1). Produces the 4D field ``M(x, y, z, t)`` that ``renderer/gas_volume.py``
raymarches above the world as sub-tile smoke detail: a fully PERIODIC box
(spatial loop for free) run long enough to become statistically stationary,
with the time loop closed by a two-period crossfade. OFFLINE and pyray-free:
numpy + scipy only, so the solver is headless-testable and the game never
imports it on the hot path (the loader in gas_volume.py reads the .npz).

Physics (incompressible Navier-Stokes, Stam's stable-fluids splitting):
  1. forces   — a divergence-free low-wavenumber body force (a few slowly
                rotating Fourier modes, each polarised perpendicular to its
                wavevector) keeps turbulence alive against semi-Lagrangian
                damping; vorticity confinement re-injects the small swirls;
                mild buoyancy on the "fire" scalar gives a vertical preference;
  2. advect   — semi-Lagrangian trilinear backtrace with the MacCormack
                correction + a min/max clamp (sharp filaments at modest grids);
  3. project  — exact on the torus via FFT: u_k -= k (k.u_k) / |k|^2;
  4. scalars  — rho (smoke) and phi (fire) advected the same way, then RELAXED
                toward slowly drifting low-frequency targets so contrast never
                washes out and the statistics are stationary.

Time loop: record 2T frames after warm-up and output
``O(k) = (1 - s) F(k + T) + s F(k)``, ``s = smoothstep(k / T)``, so
``O(0) == O(T) == F(T)`` exactly — the loop closes and the blend is invisible
in smoke.

Asset contract (``assets/volumes/<name>.npz``, read by gas_volume.py):
  volume        uint8 (T, Z, Y, X, 2) — [..., 0] smoke, [..., 1] fire
  period_tiles  float — the box's x/y extent in world tiles (texture repeat)
  height_tiles  float — the box's z extent in world tiles
  loop_seconds  float — wall/sim seconds one loop of T frames spans
  version       int   — VOLUME_FORMAT_VERSION

Credit (repo rule — cite what a file implements):
  - Jos Stam, "Stable Fluids", SIGGRAPH 1999 — the splitting.
  - Jos Stam, "A Simple Fluid Solver based on the FFT", jgt 2001 — periodic
    projection in Fourier space.
  - A. Selle, R. Fedkiw, B. Kim, Y. Liu, J. Rossignac, "An Unconditionally
    Stable MacCormack Method", J. Sci. Comput. 2008 — the advection + clamp.
  - R. Fedkiw, J. Stam, H. W. Jensen, "Visual Simulation of Smoke", SIGGRAPH
    2001 — vorticity confinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage

VOLUME_FORMAT_VERSION = 1

_Progress = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class BakeParams:
    """Everything the bake depends on. Grid axes are (z, y, x); velocities
    are in CELLS PER STEP (the solver's dt is 1 step)."""
    nx: int = 128
    ny: int = 128
    nz: int = 64                # the box is 4 x 4 x 4 TILES (see height_tiles):
                                # a short vertical period smears into streaks
                                # along the view rays when a tilted ray crosses
                                # it several times (incident 2026-10-14), so
                                # the repeat is as tall as the columns get;
                                # z texels are 2x coarser than x/y
    frames: int = 48            # output frames per loop (T)
    substeps: int = 2           # solver steps per output frame
    warmup_steps: int = 160     # steps before recording (reach stationarity)
    seed: int = 7
    # --- velocity forcing ---------------------------------------------------
    force_modes: int = 8        # number of solenoidal low-k modes
    force_kmin: int = 1         # |k| floor: >= 3 keeps the box-scale modes
                                # (the ones that read as "one blob per tile
                                # period" in the game) out of the forcing
    force_kmax: int = 2         # per-axis integer wavenumber ceiling
    force_amp: float = 0.025    # acceleration, cells/step^2 (per mode, peak)
    force_rate: float = 0.03    # phase drift, rad/step (random per mode)
    force_spectrum: float = -0.5  # mode amplitude ~ |k|^spectrum
    vort_eps: float = 0.25      # vorticity confinement strength
    buoyancy: float = 0.02      # w += buoyancy * (phi - mean phi) per step
    damping: float = 0.01       # velocity drag per step (bounds the energy)
    vel_cap: float = 2.0        # |component| clamp, cells/step (backtrace safety)
    # --- scalars (shared) ---------------------------------------------------
    scalar_maccormack: bool = False  # False = plain semi-Lagrangian (softer,
                                     # smokier); True = sharper filaments
    scalar_diffusion: float = 0.12   # per-step blend toward the 3x3x3 mean
                                     # (kills grid-scale salt)
    # --- smoke scalar (rho): target = SpectralNoise (full band, no lattice) --
    rho_relax: float = 0.02     # per-step relaxation toward the target
    rho_kmin: float = 0.5       # band floor: modes |k| <= kmin are cut (2.5 =
                                # nothing larger than a third of the box)
    rho_kmax: float = 8.0       # band limit in box wavenumbers (box / 8 = the
                                # smallest target feature; advection adds finer)
    rho_rate: float = 0.015     # target phase drift, rad/step (at kmax)
    rho_spectrum: float = -1.6  # amplitude ~ |k|^spectrum (-1.5 = fBm-like)
    # --- fire scalar (phi) --------------------------------------------------
    phi_relax: float = 0.05
    phi_kmin: float = 0.5
    phi_kmax: float = 12.0
    phi_rate: float = 0.05
    phi_spectrum: float = -1.3
    phi_sparsity: float = 0.6   # smoothstep lower edge: higher = patchier fire
    # --- asset metadata -----------------------------------------------------
    period_tiles: float = 4.0
    height_tiles: float = 4.0
    loop_seconds: float = 4.0

    def shape(self) -> Tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)


@dataclass
class BakeResult:
    volume: np.ndarray          # uint8 (T, Z, Y, X, 2)
    params: BakeParams
    stats: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {
            "period_tiles": float(self.params.period_tiles),
            "height_tiles": float(self.params.height_tiles),
            "loop_seconds": float(self.params.loop_seconds),
            "version": VOLUME_FORMAT_VERSION,
        }


# ---------------------------------------------------------------------------
# Periodic grid helpers (pure numpy, all on (z, y, x) arrays)
# ---------------------------------------------------------------------------

def _grid_coords(shape: Tuple[int, int, int]) -> np.ndarray:
    """(3, nz, ny, nx) float32 cell-centre indices, ordered (z, y, x)."""
    nz, ny, nx = shape
    zz, yy, xx = np.meshgrid(np.arange(nz, dtype=np.float32),
                             np.arange(ny, dtype=np.float32),
                             np.arange(nx, dtype=np.float32), indexing="ij")
    return np.stack([zz, yy, xx])


def wavenumbers(shape: Tuple[int, int, int]):
    """Angular wavenumbers (kz, ky, kx) broadcastable over the rfftn layout
    (last axis half-spectrum). Cell spacing is 1."""
    nz, ny, nx = shape
    kz = (2.0 * np.pi * np.fft.fftfreq(nz)).astype(np.float32)
    ky = (2.0 * np.pi * np.fft.fftfreq(ny)).astype(np.float32)
    kx = (2.0 * np.pi * np.fft.rfftfreq(nx)).astype(np.float32)
    # The Nyquist mode's derivative aliases to zero on the grid (its sign is
    # ambiguous), so derivative operators treat it as k = 0 — otherwise the
    # projected half-spectrum is no longer Hermitian-consistent on that plane
    # and irfftn silently undoes the projection there.
    if nz % 2 == 0:
        kz[nz // 2] = 0.0
    if ny % 2 == 0:
        ky[ny // 2] = 0.0
    if nx % 2 == 0:
        kx[-1] = 0.0
    return kz[:, None, None], ky[None, :, None], kx[None, None, :]


def project(u: np.ndarray, v: np.ndarray, w: np.ndarray,
            kvec=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove the divergent part of (u, v, w) on the torus — exact in Fourier
    space (Stam 2001). The k = 0 mode (mean flow) is zeroed too, so the whole
    pattern never translates (drift is a shader-side choice, not a bake one).
    """
    shape = u.shape
    if kvec is None:
        kvec = wavenumbers(shape)
    kz, ky, kx = kvec
    U = np.fft.rfftn(u)
    V = np.fft.rfftn(v)
    W = np.fft.rfftn(w)
    k2 = kx * kx + ky * ky + kz * kz
    # k = 0 at the DC term AND at every pure-Nyquist mode (wavenumbers() zeroes
    # those): k.u is 0 there too, so any non-zero divisor leaves them alone.
    k2 = np.where(k2 > 0.0, k2, 1.0)
    kdotu = (kx * U + ky * V + kz * W) / k2
    U = U - kx * kdotu
    V = V - ky * kdotu
    W = W - kz * kdotu
    U[0, 0, 0] = 0.0
    V[0, 0, 0] = 0.0
    W[0, 0, 0] = 0.0
    axes = (0, 1, 2)
    return (np.fft.irfftn(U, s=shape, axes=axes).astype(np.float32),
            np.fft.irfftn(V, s=shape, axes=axes).astype(np.float32),
            np.fft.irfftn(W, s=shape, axes=axes).astype(np.float32))


def divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Central-difference periodic divergence (the projection's gate)."""
    du = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) * 0.5
    dv = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) * 0.5
    dw = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) * 0.5
    return du + dv + dw


def _ddx(f, axis):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) * 0.5


def curl(u: np.ndarray, v: np.ndarray, w: np.ndarray):
    """Periodic central-difference vorticity (wx, wy, wz); axes (z,y,x) =
    (0,1,2)."""
    wx = _ddx(w, 1) - _ddx(v, 0)
    wy = _ddx(u, 0) - _ddx(w, 2)
    wz = _ddx(v, 2) - _ddx(u, 1)
    return wx, wy, wz


def vorticity_confinement(u, v, w, eps: float):
    """Fedkiw-Stam-Jensen 2001: f = eps * (N x omega), N = grad|omega| / |.|."""
    wx, wy, wz = curl(u, v, w)
    mag = np.sqrt(wx * wx + wy * wy + wz * wz)
    nx_ = _ddx(mag, 2)
    ny_ = _ddx(mag, 1)
    nz_ = _ddx(mag, 0)
    nrm = np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) + 1e-5
    nx_ /= nrm
    ny_ /= nrm
    nz_ /= nrm
    fx = eps * (ny_ * wz - nz_ * wy)
    fy = eps * (nz_ * wx - nx_ * wz)
    fz = eps * (nx_ * wy - ny_ * wx)
    return fx.astype(np.float32), fy.astype(np.float32), fz.astype(np.float32)


# ---------------------------------------------------------------------------
# Advection — semi-Lagrangian + MacCormack with clamp, periodic
# ---------------------------------------------------------------------------

def _backtrace(grid: np.ndarray, u, v, w, dt: float) -> np.ndarray:
    """Departure points (3, nz, ny, nx) for a trilinear lookup; wrap is the
    lookup's job (mode='grid-wrap')."""
    c = np.empty_like(grid)
    c[0] = grid[0] - dt * w
    c[1] = grid[1] - dt * v
    c[2] = grid[2] - dt * u
    return c


def _sample(f: np.ndarray, coords: np.ndarray, order: int = 1) -> np.ndarray:
    return ndimage.map_coordinates(f, coords, order=order, mode="grid-wrap",
                                   prefilter=False).astype(np.float32)


class Advector:
    """Holds the per-step departure points so every field shares one
    backtrace (and one forward trace for the MacCormack correction)."""

    def __init__(self, shape: Tuple[int, int, int]):
        self.grid = _grid_coords(shape)
        # Departure points for the current velocity; a zero velocity until
        # set_velocity() is called (identity advection), never None.
        self.back: np.ndarray = self.grid.copy()
        self.fwd: np.ndarray = self.grid.copy()
        self.back_nearest: np.ndarray = self.grid.copy()

    def set_velocity(self, u, v, w, dt: float = 1.0) -> None:
        self.back = _backtrace(self.grid, u, v, w, dt)
        self.fwd = _backtrace(self.grid, u, v, w, -dt)
        self.back_nearest = np.rint(self.back)

    def semi_lagrangian(self, f: np.ndarray) -> np.ndarray:
        return _sample(f, self.back)

    def maccormack(self, f: np.ndarray) -> np.ndarray:
        """Selle et al. 2008: f1 = A(f); f2 = A^-1(f1); f* = f1 + (f - f2)/2,
        clamped to the local min/max around the departure point (a 3x3x3
        envelope, a superset of the 8 trilinear corners) so no new extrema
        appear. Unconditionally stable, second order in smooth regions."""
        f1 = _sample(f, self.back)
        f2 = _sample(f1, self.fwd)
        out = f1 + 0.5 * (f - f2)
        lo = ndimage.minimum_filter(f, size=3, mode="wrap")
        hi = ndimage.maximum_filter(f, size=3, mode="wrap")
        lo_s = _sample(lo, self.back_nearest, order=0)
        hi_s = _sample(hi, self.back_nearest, order=0)
        return np.clip(out, lo_s, hi_s).astype(np.float32)


# ---------------------------------------------------------------------------
# Low-wavenumber drifting mode sets (forcing + scalar targets)
# ---------------------------------------------------------------------------

class ModeSet:
    """A sum of ``n`` periodic plane waves with integer wavevectors
    ``|k_i| <= kmax`` (never all zero), random phases drifting at random rates
    and amplitudes breathing on their own slow clocks. Evaluated on the grid
    at step ``t``. ``solenoidal=True`` gives each mode a unit polarisation
    perpendicular to its wavevector, so the vector field is divergence-free
    by construction (the velocity forcing)."""

    def __init__(self, rng: np.random.Generator, shape, n: int, kmax: int,
                 rate: float, amp: float, solenoidal: bool,
                 spectrum: float = -0.5, kmin: int = 1):
        nz, ny, nx = shape
        self.shape = shape
        self.solenoidal = solenoidal
        ks = []
        while len(ks) < n:
            k = rng.integers(-kmax, kmax + 1, size=3)
            kn = np.linalg.norm(k)
            if not np.any(k) or kn > kmax + 0.5 or kn < kmin - 0.5:
                continue
            ks.append(k)
        self.k = np.array(ks, dtype=np.float32)            # (n, 3) in (z,y,x)
        self.phase0 = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.rate = rng.uniform(-rate, rate, n).astype(np.float32)
        self.env_rate = rng.uniform(0.3, 1.0, n).astype(np.float32) * rate
        self.env_phase = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        kmag = np.linalg.norm(self.k, axis=1)
        # Red spectrum: amplitude ~ |k|^spectrum (-0.5 = gentle tilt for the
        # forcing; ~-1.7 for the scalar targets = fBm-like, no visible lattice).
        self.amp = (amp * np.power(kmag, spectrum)).astype(np.float32)
        if solenoidal:
            # Polarisation perpendicular to the PHYSICAL wavevector
            # 2*pi*(m_z/nz, m_y/ny, m_x/nx) — not the integer index vector,
            # which differs from it whenever the grid is not a cube.
            n_axis = np.array([nz, ny, nx], dtype=np.float64)
            pol = []
            for kk in self.k:
                kp = kk.astype(np.float64) / n_axis
                r = rng.normal(size=3)
                r -= kp * (r @ kp) / (kp @ kp)
                pol.append(r / np.linalg.norm(r))
            self.pol = np.array(pol, dtype=np.float32)       # (n, 3) in (z,y,x)
        # Angular position of every cell for each mode: 2*pi*k.x/N per axis.
        zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                                 indexing="ij")
        self._arg = [
            (2 * np.pi * (kk[0] * zz / nz + kk[1] * yy / ny + kk[2] * xx / nx)
             ).astype(np.float32) for kk in self.k]

    def scalar(self, t: float) -> np.ndarray:
        out = np.zeros(self.shape, dtype=np.float32)
        for i, arg in enumerate(self._arg):
            env = 0.5 + 0.5 * np.sin(self.env_rate[i] * t + self.env_phase[i])
            out += self.amp[i] * env * np.cos(arg + self.phase0[i]
                                              + self.rate[i] * t)
        return out

    def vector(self, t: float):
        assert self.solenoidal
        fz = np.zeros(self.shape, dtype=np.float32)
        fy = np.zeros(self.shape, dtype=np.float32)
        fx = np.zeros(self.shape, dtype=np.float32)
        for i, arg in enumerate(self._arg):
            env = 0.5 + 0.5 * np.sin(self.env_rate[i] * t + self.env_phase[i])
            c = self.amp[i] * env * np.cos(arg + self.phase0[i]
                                           + self.rate[i] * t)
            fz += self.pol[i, 0] * c
            fy += self.pol[i, 1] * c
            fx += self.pol[i, 2] * c
        return fx, fy, fz


class SpectralNoise:
    """Band-limited PERIODIC noise with EVERY mode ``0 < |k| <= kmax`` present
    (a sparse sum of plane waves shows a lattice; the full band does not),
    amplitude ``|k|^spectrum`` (-1.5 = equal energy per octave, fBm-like;
    more negative = smoother) and random phases drifting at random rates
    (faster for small scales), so the field evolves smoothly in time. One
    irfftn per evaluation. The scalar targets' generator."""

    def __init__(self, rng: np.random.Generator, shape, kmax: float,
                 rate: float, spectrum: float, kmin: float = 0.5):
        nz, ny, nx = shape
        self.shape = shape
        kz = np.fft.fftfreq(nz) * nz
        ky = np.fft.fftfreq(ny) * ny
        kx = np.fft.rfftfreq(nx) * nx
        K = np.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2
                    + kx[None, None, :] ** 2)
        mask = (K > max(float(kmin), 0.5)) & (K <= float(kmax))
        with np.errstate(divide="ignore"):
            amp = np.where(mask, np.power(np.maximum(K, 1e-6), spectrum), 0.0)
        self.amp = amp.astype(np.float32)
        half = self.amp.shape
        self.phase0 = rng.uniform(0, 2 * np.pi, half).astype(np.float32)
        sign = rng.choice([-1.0, 1.0], size=half).astype(np.float32)
        self.rate = (sign * rate * np.sqrt(np.maximum(K, 1e-6) / max(kmax, 1.0))
                     ).astype(np.float32)

    def scalar(self, t: float) -> np.ndarray:
        spec = self.amp * np.exp(1j * (self.phase0 + self.rate * float(t)))
        return np.fft.irfftn(spec, s=self.shape, axes=(0, 1, 2)).astype(np.float32)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit_target(raw: np.ndarray, lo_edge: float = -1.0,
                 hi_edge: float = 1.0) -> np.ndarray:
    """Map a zero-mean mode sum into [0, 1] with soft shoulders."""
    s = raw / (np.std(raw) + 1e-6)
    return _smoothstep(lo_edge, hi_edge, s)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class PeriodicSmokeSolver:
    """State + one ``step()``. Fields are (nz, ny, nx) float32."""

    def __init__(self, p: BakeParams):
        self.p = p
        shape = p.shape()
        self.shape = shape
        self.rng = np.random.default_rng(p.seed)
        self.u = np.zeros(shape, np.float32)
        self.v = np.zeros(shape, np.float32)
        self.w = np.zeros(shape, np.float32)
        self.force = ModeSet(self.rng, shape, p.force_modes, p.force_kmax,
                             p.force_rate, p.force_amp, solenoidal=True,
                             spectrum=p.force_spectrum, kmin=p.force_kmin)
        self.rho_target = SpectralNoise(self.rng, shape, p.rho_kmax,
                                        p.rho_rate, p.rho_spectrum,
                                        kmin=p.rho_kmin)
        self.phi_target = SpectralNoise(self.rng, shape, p.phi_kmax,
                                        p.phi_rate, p.phi_spectrum,
                                        kmin=p.phi_kmin)
        self.rho = _unit_target(self.rho_target.scalar(0.0))
        self.phi = _smoothstep(p.phi_sparsity, 1.0,
                               _unit_target(self.phi_target.scalar(0.0)))
        self.adv = Advector(shape)
        self.kvec = wavenumbers(shape)
        self.t = 0
        # a small random seed velocity so the first steps are not symmetric
        self.u += self.rng.normal(0, 0.05, shape).astype(np.float32)
        self.v += self.rng.normal(0, 0.05, shape).astype(np.float32)
        self.w += self.rng.normal(0, 0.05, shape).astype(np.float32)
        self.u, self.v, self.w = project(self.u, self.v, self.w, self.kvec)

    def step(self) -> None:
        p = self.p
        t = float(self.t)
        # 1. forces (explicit): solenoidal forcing + confinement + buoyancy.
        fx, fy, fz = self.force.vector(t)
        cx, cy, cz = vorticity_confinement(self.u, self.v, self.w, p.vort_eps)
        buoy = p.buoyancy * (self.phi - float(self.phi.mean()))
        u = (self.u + fx + cx) * (1.0 - p.damping)
        v = (self.v + fy + cy) * (1.0 - p.damping)
        w = (self.w + fz + cz + buoy) * (1.0 - p.damping)
        np.clip(u, -p.vel_cap, p.vel_cap, out=u)
        np.clip(v, -p.vel_cap, p.vel_cap, out=v)
        np.clip(w, -p.vel_cap, p.vel_cap, out=w)
        # 2. advect velocity by itself (MacCormack).
        self.adv.set_velocity(u, v, w)
        u2 = self.adv.maccormack(u)
        v2 = self.adv.maccormack(v)
        w2 = self.adv.maccormack(w)
        # 3. project onto the divergence-free space (exact, periodic).
        self.u, self.v, self.w = project(u2, v2, w2, self.kvec)
        # 4. advect the scalars with the projected velocity (semi-Lagrangian
        #    by default — softer, smokier; MacCormack for sharp filaments),
        #    then a touch of diffusion against grid-scale salt.
        self.adv.set_velocity(self.u, self.v, self.w)
        if p.scalar_maccormack:
            rho = self.adv.maccormack(self.rho)
            phi = self.adv.maccormack(self.phi)
        else:
            rho = self.adv.semi_lagrangian(self.rho)
            phi = self.adv.semi_lagrangian(self.phi)
        if p.scalar_diffusion > 0.0:
            a = float(p.scalar_diffusion)
            rho = (1.0 - a) * rho + a * ndimage.uniform_filter(rho, 3, mode="wrap")
            phi = (1.0 - a) * phi + a * ndimage.uniform_filter(phi, 3, mode="wrap")
        # 5. relax toward the drifting targets (stationary statistics).
        rho_t = _unit_target(self.rho_target.scalar(t))
        phi_t = _smoothstep(p.phi_sparsity, 1.0,
                            _unit_target(self.phi_target.scalar(t)))
        rho += p.rho_relax * (rho_t - rho)
        phi += p.phi_relax * (phi_t - phi)
        self.rho = np.clip(rho, 0.0, 1.0).astype(np.float32)
        self.phi = np.clip(phi, 0.0, 1.0).astype(np.float32)
        self.t += 1

    def frame(self) -> np.ndarray:
        """(Z, Y, X, 2) float32 snapshot (smoke, fire)."""
        return np.stack([self.rho, self.phi], axis=-1)


# ---------------------------------------------------------------------------
# Loop closure + quantisation
# ---------------------------------------------------------------------------

def close_loop(frames: np.ndarray) -> np.ndarray:
    """Crossfade 2T recorded frames into T looping ones.

    ``frames`` is (2T, ...) ; returns (T, ...) with
    ``O(k) = (1 - s) F(k + T) + s F(k)``, ``s = smoothstep(k / T)``. The
    property that matters: ``O(0) == F(T)`` and the frame AFTER ``O(T-1)`` in
    the loop is ``O(0)``, one solver step on from ``F(T-1)`` — no seam."""
    n2 = frames.shape[0]
    assert n2 % 2 == 0, "need 2T frames"
    T = n2 // 2
    out = np.empty((T,) + frames.shape[1:], dtype=np.float32)
    for k in range(T):
        s = k / T
        s = s * s * (3.0 - 2.0 * s)
        out[k] = (1.0 - s) * frames[k + T] + s * frames[k]
    return out


def quantize(volume_f: np.ndarray, lo_pct: float = 0.5,
             hi_pct: float = 99.7) -> Tuple[np.ndarray, dict]:
    """Per-channel percentile stretch to uint8 (the shader reads [0,1])."""
    out = np.empty(volume_f.shape, dtype=np.uint8)
    stats = {}
    for c in range(volume_f.shape[-1]):
        ch = volume_f[..., c]
        lo = float(np.percentile(ch, lo_pct))
        hi = float(np.percentile(ch, hi_pct))
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        out[..., c] = np.clip((ch - lo) / (hi - lo) * 255.0 + 0.5,
                              0, 255).astype(np.uint8)
        stats[f"ch{c}_lo"] = lo
        stats[f"ch{c}_hi"] = hi
    return out, stats


# ---------------------------------------------------------------------------
# The bake
# ---------------------------------------------------------------------------

def bake(p: BakeParams, progress: _Progress = None) -> BakeResult:
    """Warm up, record 2T frames, close the loop, quantise."""
    say = progress or (lambda _msg: None)
    solver = PeriodicSmokeSolver(p)
    say(f"[bake] grid {p.nx}x{p.ny}x{p.nz}, {p.frames} frames x "
        f"{p.substeps} substeps, warm-up {p.warmup_steps}")
    for i in range(p.warmup_steps):
        solver.step()
        if (i + 1) % 20 == 0:
            say(f"[bake] warm-up {i + 1}/{p.warmup_steps}  "
                f"|u|max={max(np.abs(solver.u).max(), np.abs(solver.v).max(), np.abs(solver.w).max()):.2f}  "
                f"rho std={solver.rho.std():.3f}")
    n_rec = 2 * p.frames
    # float16 recording: a 256x256x64 x 2T x 2 bake is 1.6 GB here instead
    # of 3.2 (the fields are [0,1]; half precision is far below the uint8
    # the asset ends up as).
    rec = np.empty((n_rec,) + p.shape() + (2,), dtype=np.float16)
    for k in range(n_rec):
        for _ in range(p.substeps):
            solver.step()
        rec[k] = solver.frame()
        if (k + 1) % 8 == 0:
            say(f"[bake] record {k + 1}/{n_rec}")
    looped = close_loop(rec)
    vol, stats = quantize(looped)
    stats["div_max"] = float(np.abs(divergence(solver.u, solver.v,
                                               solver.w)).max())
    say(f"[bake] done: {vol.shape} uint8, stats {stats}")
    return BakeResult(volume=vol, params=p, stats=stats)


def procedural_volume(frames: int = 16, nz: int = 16, ny: int = 64,
                      nx: int = 64, seed: int = 3) -> np.ndarray:
    """A FAST looping, tiling fallback volume (seconds, no solver): a few
    periodic plane waves whose phases advance by whole turns over the loop.
    Used when no baked asset exists so a fresh clone still runs, and by
    tests that need a valid volume without a bake."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    out = np.zeros((frames, nz, ny, nx, 2), dtype=np.float32)
    for c in range(2):
        n_modes = 12 if c == 0 else 16
        kmax = 3 if c == 0 else 5
        acc = np.zeros((frames, nz, ny, nx), dtype=np.float32)
        for _ in range(n_modes):
            k = rng.integers(-kmax, kmax + 1, size=3)
            if not np.any(k):
                continue
            turns = int(rng.integers(-2, 3))          # whole turns -> loops
            ph0 = rng.uniform(0, 2 * np.pi)
            amp = 1.0 / np.sqrt(float(k @ k))
            arg = 2 * np.pi * (k[0] * zz / nz + k[1] * yy / ny + k[2] * xx / nx)
            for t in range(frames):
                acc[t] += amp * np.cos(arg + ph0 + 2 * np.pi * turns * t / frames)
        out[..., c] = _unit_target(acc, -0.8, 1.2)
    vol, _ = quantize(out)
    return vol


# ---------------------------------------------------------------------------
# Preview + I/O (pyray-free; PIL only for the mosaic)
# ---------------------------------------------------------------------------

def mosaic(volume: np.ndarray, channel: int = 0, cols: int = 8,
           z_index: Optional[int] = None) -> np.ndarray:
    """(T, Z, Y, X, C) uint8 -> a (rows*Y, cols*X) uint8 mosaic of one
    z-slice across frames (``z_index`` None = middle slice). For eyeballing
    the bake before it reaches the GPU."""
    T, Z, Y, X, _ = volume.shape
    zi = Z // 2 if z_index is None else int(z_index)
    rows = (T + cols - 1) // cols
    img = np.zeros((rows * Y, cols * X), dtype=np.uint8)
    for t in range(T):
        r, c = divmod(t, cols)
        img[r * Y:(r + 1) * Y, c * X:(c + 1) * X] = volume[t, zi, :, :, channel]
    return img


def mosaic_z(volume: np.ndarray, channel: int = 0, cols: int = 8,
             frame: int = 0) -> np.ndarray:
    """All z-slices of one frame (the vertical structure)."""
    _T, Z, Y, X, _ = volume.shape
    rows = (Z + cols - 1) // cols
    img = np.zeros((rows * Y, cols * X), dtype=np.uint8)
    for z in range(Z):
        r, c = divmod(z, cols)
        img[r * Y:(r + 1) * Y, c * X:(c + 1) * X] = volume[frame, z, :, :, channel]
    return img


def save_volume(path, result: BakeResult) -> None:
    np.savez_compressed(path, volume=result.volume, **result.meta(),
                        params=np.array([repr(asdict(result.params))]))


def load_volume(path) -> Tuple[np.ndarray, dict]:
    """-> (volume uint8 (T,Z,Y,X,2), meta dict). Raises on a wrong format."""
    with np.load(path, allow_pickle=False) as z:
        vol = np.asarray(z["volume"])
        meta = {
            "period_tiles": float(z["period_tiles"]),
            "height_tiles": float(z["height_tiles"]),
            "loop_seconds": float(z["loop_seconds"]),
            "version": int(z["version"]),
        }
    if meta["version"] != VOLUME_FORMAT_VERSION:
        raise ValueError(f"volume format {meta['version']} != "
                         f"{VOLUME_FORMAT_VERSION}")
    if vol.ndim != 5 or vol.shape[-1] != 2 or vol.dtype != np.uint8:
        raise ValueError(f"bad volume array {vol.shape} {vol.dtype}")
    return vol, meta


__all__ = [
    "VOLUME_FORMAT_VERSION", "BakeParams", "BakeResult",
    "PeriodicSmokeSolver", "Advector", "ModeSet", "SpectralNoise",
    "wavenumbers", "project", "divergence", "curl", "vorticity_confinement",
    "close_loop", "quantize", "bake", "procedural_volume",
    "mosaic", "mosaic_z", "save_volume", "load_volume",
]
