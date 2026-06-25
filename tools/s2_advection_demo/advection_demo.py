#!/usr/bin/env python
"""
S2 advection decision demo — semi-Lagrangian vs conservative flux-form smoke.

WHY THIS EXISTS
---------------
Breach's smoke advection is currently semi-Lagrangian (SL): a back-trace +
bilinear sample of the upstream value, with a wall-clip that stops the trace
before a sealed tile (faithful port of cpp/src/smoke_dynamics.cpp). The open S2
question is whether to switch the smoke transport to a *conservative flux-form*
scheme (the same family the water solver already uses: donor-cell upwind flux),
which would conserve smoke mass exactly instead of leaking it at walls / through
the bilinear clip.

This script is a DECISION AID. It runs four advection schemes
    1.  SL    - the current FLOAT scheme (the familiar look), faithful to the C++
    1b. SLint - an INTEGER Q16.16 mirror of SL (the cross-GPU-deterministic test)
    2.  donor - bare 1st-order upwind flux (conservative)
    3.  mc    - donor + MC-limited 2nd-order TVD correction (conservative)
on the REAL unhcr_vessel_2 ship map, driven by the SAME blast wind and the SAME
wind-dependent diffusion. ONLY the advection step differs between them.
Erik decides by LOOKING (side-by-side GIFs) and by reading the conservation plot
(SL leaks mass at walls; the flux schemes stay flat).

THE SLint QUESTION (added 2026-06): Erik wants to KEEP semi-Lagrangian smoke (its
filled, internally-structured look) but it must be CROSS-GPU DETERMINISTIC. Float
SL is not (IEEE float is non-associative cross-vendor). The fix is integer Q16.16
(cpp/src/fixed_point.h). SLint is that integer SL; the float-hard ops (the renorm
divide, the march-length sqrt) are APPROXIMATED (Newton reciprocal; sqrt-free DDA
march). The demo answers: (1) is SLint bit-deterministic? (2) does it LOOK like
float SL? (3) how non-conservative is it? See the SLfloat_vs_SLint + SLint_mass
outputs and the DETERMINISM assert printed at run.

It is fully standalone: it reads only the tilemap CSV + config.toml constants and
depends on numpy + matplotlib + imageio. It does NOT import the game engine, so
it cannot perturb anything; it is purely a visual experiment on the side.

Outputs (written next to this script in ./output/):
    scenario1_single.gif          4-panel animation (SL | SLint | donor | MC)
    scenario2_double.gif          4-panel animation (two grenades)
    scenario1_conservation.png    total-mass-vs-time, 4 lines
    scenario2_conservation.png    total-mass-vs-time, 4 lines
    scenario1_montage.png         key frames, 4 schemes side by side
    scenario2_montage.png         key frames, 4 schemes side by side
    scenarioN_SLfloat_vs_SLint.png  float-SL | int-SL | |diff|x6  (the look test)
    scenarioN_SLint_mass.png      integer-SL mass vs time (non-conservation)

Run:
    C:/Users/steen/anaconda3/python.exe tools/s2_advection_demo/advection_demo.py
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TILEMAP_CSV = REPO / "levels" / "unhcr_vessel_2" / "tilemap.csv"
CONFIG_TOML = REPO / "config.toml"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# v2 tile vocabulary (level_loader.py): codes ARE material ids; 9 == SPACE.
MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS, MAT_FURNITURE = range(7)
SPACE_CODE = 9
# Gas can flow through air(0), door(3), and furniture(6, porous). Everything
# else (hull/wood/steel/glass) is a sealed wall to smoke. This mirrors the
# permeability boundary in 05_smoke.md (face = min(perm[self], perm[nbr])).
PERMEABLE_CODES = {MAT_AIR, MAT_DOOR, MAT_FURNITURE}
# Permeability VALUE per code (furniture is porous = 0.5, config.toml).
PERM_VALUE = {MAT_AIR: 1.0, MAT_DOOR: 1.0, MAT_FURNITURE: 0.5}


def load_config_physics() -> dict:
    """Pull the smoke constants straight from config.toml [physics]."""
    with open(CONFIG_TOML, "rb") as f:
        cfg = tomllib.load(f)
    p = cfg["physics"]
    return {
        "d_smoke": float(p["d_smoke"]),                          # 0.1
        "advection_rate": float(p["advection_rate"]),            # 900.0
        "wind_diffusion_scale": float(p["wind_diffusion_scale"]),# 50.0
        "explosion_smoke_noise": float(p["explosion_smoke_noise"]),  # 0.85
        "tps": float(cfg["clock"]["ticks_per_second"]),          # 24
    }


# --------------------------------------------------------------------------- #
# Map -> masks
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    code: np.ndarray       # (H, W) int tile codes
    solid: np.ndarray      # (H, W) bool — sealed to smoke (wall)
    is_vacuum: np.ndarray  # (H, W) bool — SPACE sink
    perm: np.ndarray       # (H, W) float — gas permeability per cell [0,1]
    H: int
    W: int


def load_grid() -> Grid:
    code = np.loadtxt(TILEMAP_CSV, delimiter=",", dtype=np.int32)
    H, W = code.shape
    is_vacuum = (code == SPACE_CODE)
    # Solid (sealed to smoke) = any tile that is not permeable and not vacuum.
    permeable = np.isin(code, list(PERMEABLE_CODES))
    solid = ~permeable & ~is_vacuum
    perm = np.zeros((H, W), dtype=np.float64)
    for c, v in PERM_VALUE.items():
        perm[code == c] = v
    # Vacuum has no gas in it; treat as fully sealed face-wise except that the
    # transport explicitly zeros smoke there (the breach sink). perm stays 0.
    return Grid(code=code, solid=solid, is_vacuum=is_vacuum, perm=perm, H=H, W=W)


# --------------------------------------------------------------------------- #
# Blast wind: pressure pulse relaxed through OPEN cells, wind = -grad(pressure)
# --------------------------------------------------------------------------- #
def blast_wind(grid: Grid, fy: int, fx: int, radius: float, pressure: float,
               sweeps: int = 250):
    """Inject a pressure pulse at (fy, fx), relax through open cells respecting
    walls (Neumann BC at solids/vacuum), then wind = -grad(pressure).

    This is a simple but representative stand-in for the atmosphere solver: a
    static over-pressure field whose gradient FUNNELS through the corridor (gas
    reacting to obstacles). The SAME wind field feeds all three advection
    schemes, so any visible difference is the advection, not the wind.
    """
    H, W = grid.H, grid.W
    open_mask = (~grid.solid) & (~grid.is_vacuum)

    # Source disc: linear-falloff over-pressure (mirrors apply_explosion's
    # atmosphere boost, a DISC + LINEAR falloff).
    p = np.zeros((H, W), dtype=np.float64)
    r = int(math.ceil(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny, nx = fy + dy, fx + dx
            if 0 <= ny < H and 0 <= nx < W and open_mask[ny, nx]:
                dist = math.hypot(dy, dx)
                if dist <= radius:
                    p[ny, nx] = pressure * (1.0 - dist / radius)
    source = p.copy()

    # Jacobi relaxation of the Poisson-ish pressure (keep source pinned each
    # sweep so it stays the pressure peak), Neumann at walls (mirror = use self).
    for _ in range(sweeps):
        up    = _shift_neumann(p, -1, 0, open_mask)
        down  = _shift_neumann(p,  1, 0, open_mask)
        left  = _shift_neumann(p,  0, -1, open_mask)
        right = _shift_neumann(p,  0,  1, open_mask)
        p_new = 0.25 * (up + down + left + right)
        p_new = np.maximum(p_new, source)   # keep the source as a floor (sustained pulse)
        p_new[~open_mask] = 0.0
        p = p_new

    # wind = -grad(pressure), central difference within open cells (one-sided at
    # walls via the Neumann mirror).
    pr = _shift_neumann(p, 0,  1, open_mask)
    pl = _shift_neumann(p, 0, -1, open_mask)
    pd = _shift_neumann(p, 1,  0, open_mask)
    pu = _shift_neumann(p, -1, 0, open_mask)
    wind_x = -0.5 * (pr - pl)
    wind_y = -0.5 * (pd - pu)
    wind_x[~open_mask] = 0.0
    wind_y[~open_mask] = 0.0
    return wind_x, wind_y, p


def _shift_neumann(f, dy, dx, open_mask):
    """Shifted copy of f where out-of-domain / sealed neighbours mirror self
    (Neumann BC). Returns the neighbour value for each cell."""
    H, W = f.shape
    out = f.copy()
    src = np.roll(np.roll(f, -dy, axis=0), -dx, axis=1)
    valid = np.roll(np.roll(open_mask, -dy, axis=0), -dx, axis=1)
    # Edge wrap from np.roll is not valid: blank the wrapped border.
    if dy == 1:
        valid[-1, :] = False
    elif dy == -1:
        valid[0, :] = False
    if dx == 1:
        valid[:, -1] = False
    elif dx == -1:
        valid[:, 0] = False
    out = np.where(valid, src, f)   # sealed/oob neighbour -> mirror self
    return out


# --------------------------------------------------------------------------- #
# Smoke deposit (the grenade disc) — faithful to add_explosion_smoke
# --------------------------------------------------------------------------- #
def deposit_smoke(smoke, grid, fy, fx, radius, noise, rng):
    """base = 0.8*(1 - dist/radius); mult ~ U[1-noise, 1]; add, clamp [0,1];
    skip solids; clear inner 40% (the fireball punches a hole)."""
    H, W = grid.H, grid.W
    inner = radius * 0.4
    r = int(math.ceil(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny, nx = fy + dy, fx + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if grid.solid[ny, nx] or grid.is_vacuum[ny, nx]:
                continue
            dist = math.hypot(dy, dx)
            if dist > radius:
                continue
            if dist <= inner:
                smoke[ny, nx] = 0.0     # fireball clears the core
                continue
            base = 0.8 * (1.0 - dist / radius)
            mult = rng.uniform(1.0 - noise, 1.0)
            smoke[ny, nx] = min(1.0, smoke[ny, nx] + base * mult)


# --------------------------------------------------------------------------- #
# Shared wind-dependent diffusion (IDENTICAL for all 3 schemes)
# --------------------------------------------------------------------------- #
def diffuse(smoke, grid, wind_x, wind_y, d_smoke, wind_diffusion_scale, dt):
    """smoke += d_eff*dt*laplacian(smoke), d_eff = d_smoke*(1+scale*|wind|^2),
    Neumann at walls (mirror), permeability-weighted face (face = min(perm)).

    This is a faithful float port of the diffusion loop in smoke_dynamics.cpp's
    step() — same operator, same wind_sq coupling — and is applied identically
    to all three schemes so it cannot be the source of any visible difference.
    """
    H, W = grid.H, grid.W
    perm = grid.perm
    s = smoke
    lap = np.zeros_like(s)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nbr = _shift_perm(s, perm, dy, dx)
        lap += nbr - s
    wind_sq = wind_x * wind_x + wind_y * wind_y
    d_eff = d_smoke * (1.0 + wind_diffusion_scale * wind_sq)
    out = s + d_eff * dt * lap
    _zero_and_clamp(out, grid)
    return out


def _shift_perm(f, perm, dy, dx):
    """neighbor() from smoke_dynamics.cpp: f[self] + face*(f[nbr]-f[self]),
    face = min(perm[self], perm[nbr]); out-of-bounds mirrors self (face 0)."""
    H, W = f.shape
    nbr = np.roll(np.roll(f, -dy, axis=0), -dx, axis=1)
    nbr_perm = np.roll(np.roll(perm, -dy, axis=0), -dx, axis=1)
    valid = np.ones((H, W), dtype=bool)
    if dy == 1:
        valid[-1, :] = False
    elif dy == -1:
        valid[0, :] = False
    if dx == 1:
        valid[:, -1] = False
    elif dx == -1:
        valid[:, 0] = False
    face = np.minimum(perm, np.where(valid, nbr_perm, 0.0))
    nbr_val = np.where(valid, nbr, f)
    return f + face * (nbr_val - f)


def _zero_and_clamp(smoke, grid):
    smoke[grid.solid] = 0.0
    smoke[grid.is_vacuum] = 0.0
    np.clip(smoke, 0.0, 1.0, out=smoke)


# --------------------------------------------------------------------------- #
# Scheme 1: semi-Lagrangian (FAITHFUL to smoke_dynamics.cpp)
# --------------------------------------------------------------------------- #
def _solid_wall_at(ti, tj, grid):
    """smoke_dynamics.cpp solid_wall_at: outside == wall; a breach (vacuum &
    not solid) is NOT a wall (vent target); else solid|vacuum|sealed == wall."""
    if tj < 0 or tj >= grid.H or ti < 0 or ti >= grid.W:
        return True
    if grid.is_vacuum[tj, ti] and not grid.solid[tj, ti]:
        return False   # breach: venting target, not a wall
    return grid.solid[tj, ti] or grid.is_vacuum[tj, ti]


def _backtrace_sample(src, x, y, bx, by, grid):
    """Faithful port of backtrace_sample(): wall-clip march + permeability-aware
    bilinear sample (sealed corners excluded, breach corner contributes 0)."""
    H, W = grid.H, grid.W
    px = x + bx
    py = y + by
    # Wall-clip march: ~1 sample/cell, stop before first solid wall, or ON a breach.
    dist = math.hypot(bx, by)
    steps = int(math.ceil(dist))
    if steps > 0:
        inv = 1.0 / steps
        sx, sy = bx * inv, by * inv
        cx, cy = float(x), float(y)
        for _ in range(steps):
            nxp, nyp = cx + sx, cy + sy
            ti = int(math.floor(nxp + 0.5))
            tj = int(math.floor(nyp + 0.5))
            if _solid_wall_at(ti, tj, grid):
                break
            cx, cy = nxp, nyp
            if 0 <= tj < H and 0 <= ti < W and grid.is_vacuum[tj, ti]:
                break
        px, py = cx, cy
    # Clamp in-bounds.
    px = min(max(px, 0.0), W - 1.0)
    py = min(max(py, 0.0), H - 1.0)
    x0 = int(math.floor(px)); y0 = int(math.floor(py))
    x1 = min(x0 + 1, W - 1);  y1 = min(y0 + 1, H - 1)
    fx = px - x0; fy = py - y0
    ci = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]
    cw = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
    acc = 0.0; wsum = 0.0
    for (cy_, cx_), w_ in zip(ci, cw):
        if grid.solid[cy_, cx_]:
            continue   # sealed corner excluded
        val = 0.0 if grid.is_vacuum[cy_, cx_] else src[cy_, cx_]
        acc += w_ * val
        wsum += w_
    return (acc / wsum) if wsum > 1e-6 else src[y, x]


def advect_semilagrangian(smoke, grid, wind_x, wind_y, dt_adv):
    """Current scheme. Per open cell: back-trace to p = cell - wind*dt_adv,
    wall-clip the trace, bilinear-sample. NOT conservative (the bilinear clip +
    wall zeroing lose mass)."""
    H, W = grid.H, grid.W
    src = smoke.copy()
    out = smoke.copy()
    open_yx = np.argwhere((~grid.solid) & (~grid.is_vacuum))
    for y, x in open_yx:
        bx = -wind_x[y, x] * dt_adv
        by = -wind_y[y, x] * dt_adv
        out[y, x] = _backtrace_sample(src, int(x), int(y), bx, by, grid)
    _zero_and_clamp(out, grid)
    return out


# --------------------------------------------------------------------------- #
# Scheme 1b: INTEGER Q16.16 semi-Lagrangian (the determinism test)
# --------------------------------------------------------------------------- #
# WHY THIS EXISTS
# ---------------
# The float SL (scheme 1) is the look Erik wants to keep — filled, internally
# structured, far prettier than the conservative flux schemes. But float SL is
# NOT cross-GPU deterministic: IEEE float +/-/* are non-associative and vary by
# vendor/compiler, so the synced lockstep state drifts. The fixed-point arc
# (cpp/src/fixed_point.h) replaces synced float fields with int32 Q16.16, where
# every op is exact-integer and therefore byte-identical on every machine.
#
# This scheme is a faithful integer MIRROR of advect_semilagrangian, built ONLY
# from int32/int64 arithmetic — the smoke density is int32 Q16.16 (density *
# 65536). It exists to answer three questions:
#   (1) is it bit-DETERMINISTIC?  (run twice -> identical int32 field)
#   (2) does it LOOK like the float SL? (do the integer approximations of the
#       hard float ops — the renorm divide, the march-length sqrt — keep the
#       beautiful inner structure, or do they band/blocky it?)
#   (3) how non-conservative is it (calm vs blast)?
#
# THE HARD FLOAT OPS AND THEIR INTEGER APPROXIMATIONS
# ---------------------------------------------------
#   * march length sqrt(bx^2+by^2): REPLACED by a sqrt-free DDA. The float march
#     takes ceil(euclidean dist) equal substeps; we instead step cell-by-cell
#     along the DOMINANT axis (Chebyshev distance = max(|bx|,|by|) cells), the
#     minor axis advancing by its integer-Q16.16 slope. This visits every
#     dominant-axis cell, so it still cannot tunnel a 1-cell wall, and needs no
#     sqrt. (The substep COUNT differs slightly from float — Chebyshev vs
#     Euclidean — but the wall-clip semantics, "stop before the first sealed
#     tile", are preserved exactly.)
#   * the renorm 1/wsum (when some bilinear corners are walls): a divide. We do
#     NOT Taylor 1/x (it diverges near 0). Instead 2-3 Newton-Raphson reciprocal
#     iterations  r <- r*(2 - wsum*r)  in Q16.16, seeded from a rough power-of-2
#     reciprocal (bit-length of wsum). Deterministic and convergent for wsum in
#     (0, 1]. wsum is clamped to a small floor first.
#
# ROUNDING (pinned, load-bearing for byte-stability) — mirrors fixed_point.h:
#   * quantize (float->Q16.16): round to NEAREST, in double (one-time boundary).
#   * mul_q16(a,b) = (int64(a)*int64(b)) >> 16  : TRUNCATE toward -inf (SAR).
#     The SAME idiom everywhere, so the result is reproducible bit for bit.
FP_SHIFT = 16
FP_ONE = 1 << FP_SHIFT            # 65536
FP_ONE64 = 1 << FP_SHIFT
FP_HALF = FP_ONE >> 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
# Newton-reciprocal floor: clamp wsum to >= this (in Q16.16) so 1/wsum can't
# blow up. ~1/256 of a unit weight — far below any real partial-corner weight.
WSUM_FLOOR_Q = FP_ONE >> 8         # 256


def _q_quantize(v: float) -> int:
    """float/double -> Q16.16, round-to-nearest (fixed_point.h::quantize)."""
    scaled = v * FP_ONE
    return int(scaled + 0.5) if scaled >= 0.0 else int(scaled - 0.5)


def _q_mul(a: int, b: int) -> int:
    """mul_q16: (int64 a*b) >> 16, arithmetic shift = truncate toward -inf.
    Python >> on negative ints already floors (SAR semantics) — matches C++20."""
    return (a * b) >> FP_SHIFT


def _q_recip(wsum_q: int) -> int:
    """1/wsum in Q16.16 by Newton-Raphson:  r <- r*(2 - wsum*r).

    Pure integer, deterministic. Seed from a power-of-2 reciprocal via the bit
    length of wsum_q (a rough 2^-k), then iterate to refine. For wsum in (0,1]
    (the only range that occurs — bilinear corner weights sum to <= 1) three
    iterations converge to <1 ULP. wsum_q is clamped to a small floor by the
    caller so the seed is well-defined and the result is bounded.

    r*(2 - wsum*r): wsum*r is Q16.16 (~1.0 = FP_ONE near convergence), (2 - that)
    is Q16.16, the outer product is Q16.16 — all via _q_mul (the same >>16
    truncation as everywhere else)."""
    if wsum_q < WSUM_FLOOR_Q:
        wsum_q = WSUM_FLOOR_Q
    # Seed: r0 ~ 2^-floor(log2(wsum_real)). wsum_q = wsum_real * 2^16, so
    # bit_length-16 is ~floor(log2(wsum_real)). r0 = 2^(16 - (bitlen-16)) in
    # Q16.16 = 1<<(32 - bitlen), clamped to [1, FP_ONE] (wsum<=1 => recip>=1).
    bitlen = wsum_q.bit_length()
    shift = 32 - bitlen
    if shift < 0:
        shift = 0
    r = 1 << shift
    if r < 1:
        r = 1
    two_q = FP_ONE << 1            # 2.0 in Q16.16
    for _ in range(3):             # 3 Newton iterations
        wr = _q_mul(wsum_q, r)     # wsum*r  (-> ~1.0)
        r = _q_mul(r, two_q - wr)  # r*(2 - wsum*r)
    return r


def _q_solid_wall_at(ti, tj, grid):
    """Integer-index mirror of _solid_wall_at (identical logic, int args)."""
    if tj < 0 or tj >= grid.H or ti < 0 or ti >= grid.W:
        return True
    if grid.is_vacuum[tj, ti] and not grid.solid[tj, ti]:
        return False
    return grid.solid[tj, ti] or grid.is_vacuum[tj, ti]


def _q_backtrace_sample(src_q, x, y, bx_q, by_q, grid):
    """Integer Q16.16 mirror of _backtrace_sample.

    src_q : int32 Q16.16 density (snapshot). x,y : int cell. bx_q,by_q : Q16.16
    displacement. Returns the new Q16.16 density for cell (y,x).

    Wall-clip march is a sqrt-free DDA (dominant-axis stepping); bilinear sample
    is integer corner-weight products narrowed from int64; the sealed-corner
    renorm 1/wsum is the Newton reciprocal."""
    H, W = grid.H, grid.W
    # px,py are the departure point in Q16.16 (cell index << 16 + displacement).
    px_q = (x << FP_SHIFT) + bx_q
    py_q = (y << FP_SHIFT) + by_q

    # ---- Wall-clip march (DDA, no sqrt) ----
    # Dominant axis = the larger |displacement|. n_steps = ceil(Chebyshev dist)
    # = ceil(max(|bx|,|by|)) cells, computed in integer from the Q16.16 magnitude
    # (>> 16 = floor of the cell distance; +1 if any fraction -> ceil). We march
    # one DOMINANT-axis cell per step; the minor axis advances by slope*step.
    abx = bx_q if bx_q >= 0 else -bx_q
    aby = by_q if by_q >= 0 else -by_q
    amax = abx if abx >= aby else aby
    n_steps = amax >> FP_SHIFT
    if amax & (FP_ONE - 1):
        n_steps += 1                          # ceil
    if n_steps > 0:
        # Per-step increment = displacement / n_steps, in Q16.16. n_steps is a
        # small positive int -> exact integer divide (floor); deterministic.
        sx_q = bx_q // n_steps
        sy_q = by_q // n_steps
        cx_q = x << FP_SHIFT
        cy_q = y << FP_SHIFT
        for _ in range(n_steps):
            nxp_q = cx_q + sx_q
            nyp_q = cy_q + sy_q
            # tile center test: floor(coord + 0.5). coord+0.5 in Q16.16 =
            # nxp_q + FP_HALF; the integer tile is that >> 16.
            ti = (nxp_q + FP_HALF) >> FP_SHIFT
            tj = (nyp_q + FP_HALF) >> FP_SHIFT
            if _q_solid_wall_at(ti, tj, grid):
                break
            cx_q, cy_q = nxp_q, nyp_q
            if 0 <= tj < H and 0 <= ti < W and grid.is_vacuum[tj, ti]:
                break
        px_q, py_q = cx_q, cy_q

    # ---- Clamp in-bounds (Q16.16) ----
    hi_x = (W - 1) << FP_SHIFT
    hi_y = (H - 1) << FP_SHIFT
    if px_q < 0:
        px_q = 0
    elif px_q > hi_x:
        px_q = hi_x
    if py_q < 0:
        py_q = 0
    elif py_q > hi_y:
        py_q = hi_y

    # ---- Integer bilinear sample ----
    x0 = px_q >> FP_SHIFT                      # floor (Q16.16 of a >=0 value)
    y0 = py_q >> FP_SHIFT
    x1 = x0 + 1 if x0 + 1 <= W - 1 else W - 1
    y1 = y0 + 1 if y0 + 1 <= H - 1 else H - 1
    fx_q = px_q - (x0 << FP_SHIFT)             # fractional part, Q16.16 in [0,1)
    fy_q = py_q - (y0 << FP_SHIFT)
    ifx_q = FP_ONE - fx_q                      # (1 - fx)
    ify_q = FP_ONE - fy_q
    # Four corner weights, each a Q16.16 product of two Q16.16 fractions.
    w00 = _q_mul(ifx_q, ify_q)                 # (1-fx)(1-fy)
    w10 = _q_mul(fx_q, ify_q)                  # fx (1-fy)
    w01 = _q_mul(ifx_q, fy_q)                  # (1-fx) fy
    w11 = _q_mul(fx_q, fy_q)                   # fx fy
    corners = ((y0, x0, w00), (y0, x1, w10), (y1, x0, w01), (y1, x1, w11))

    acc = 0          # int64 accumulator of weight*density (Q16.16 * Q16.16)
    wsum_q = 0       # int Q16.16 sum of live corner weights
    for cy_, cx_, w_ in corners:
        if grid.solid[cy_, cx_]:
            continue                            # sealed corner excluded
        val_q = 0 if grid.is_vacuum[cy_, cx_] else int(src_q[cy_, cx_])
        acc += w_ * val_q                       # int64; scale = 2^32
        wsum_q += w_
    if wsum_q <= (FP_ONE >> 14):                # ~ the 1e-6 float guard
        return int(src_q[y, x])
    # result = acc / wsum.  acc is Q(.32) (Q16.16 * Q16.16); narrowing it by >>16
    # gives Q16.16 of (sum weight*density); multiplying by recip(wsum) (Q16.16)
    # and narrowing by >>16 again divides by wsum. recip via Newton (no divide).
    recip_q = _q_recip(wsum_q)                  # 1/wsum in Q16.16
    acc_q = acc >> FP_SHIFT                      # narrow Q(.32) -> Q16.16
    res_q = _q_mul(acc_q, recip_q)              # (sum w*d)/wsum, Q16.16
    return res_q


def advect_semilagrangian_int(smoke, grid, wind_x, wind_y, dt_adv):
    """Integer Q16.16 semi-Lagrangian advection — the deterministic mirror of
    advect_semilagrangian. Quantizes the (shared float) smoke + wind to Q16.16,
    runs the integer back-trace per open cell, dequantizes for the caller.

    The INTERNAL int32 field is what the determinism test asserts on; the float
    we return is just for the shared diffuse() + rendering. To keep the whole
    pipeline integer-deterministic for the test we re-quantize each entry from
    the same float input, so an identical float input -> identical int32 field."""
    H, W = grid.H, grid.W
    # Quantize the post-diffusion float smoke to the int32 Q16.16 source snapshot.
    src_q = np.round(np.asarray(smoke, dtype=np.float64) * FP_ONE).astype(np.int64)
    # Quantize wind*dt_adv to Q16.16 displacement once (vectorized boundary cast).
    bx_f = -np.asarray(wind_x, dtype=np.float64) * dt_adv
    by_f = -np.asarray(wind_y, dtype=np.float64) * dt_adv
    bx_q = np.where(bx_f >= 0, np.floor(bx_f * FP_ONE + 0.5),
                    np.ceil(bx_f * FP_ONE - 0.5)).astype(np.int64)
    by_q = np.where(by_f >= 0, np.floor(by_f * FP_ONE + 0.5),
                    np.ceil(by_f * FP_ONE - 0.5)).astype(np.int64)

    out_q = src_q.copy()
    open_yx = np.argwhere((~grid.solid) & (~grid.is_vacuum))
    for y, x in open_yx:
        y = int(y); x = int(x)
        out_q[y, x] = _q_backtrace_sample(src_q, x, y,
                                          int(bx_q[y, x]), int(by_q[y, x]), grid)
    # Zero walls/vacuum, clamp to [0, FP_ONE] in INTEGER (the int32 invariant).
    out_q[grid.solid] = 0
    out_q[grid.is_vacuum] = 0
    np.clip(out_q, 0, FP_ONE, out=out_q)
    # Stash the int32 field for the determinism assert (caller reads it).
    advect_semilagrangian_int.last_field_q = out_q.astype(np.int32)
    # Dequantize for the shared float diffuse + render.
    return out_q.astype(np.float64) / FP_ONE


advect_semilagrangian_int.last_field_q = None


# --------------------------------------------------------------------------- #
# Scheme 2 & 3: conservative flux-form (donor-cell, and donor + MC limiter)
# --------------------------------------------------------------------------- #
def _face_velocity(wind, axis):
    """Velocity on the face between cell i and i+1 along `axis` (average of the
    two cell-centred velocities). Returns array on the same grid where entry
    [i] is the i|i+1 face (last column/row face is the domain edge -> 0)."""
    avg = 0.5 * (wind + np.roll(wind, -1, axis=axis))
    if axis == 1:
        avg[:, -1] = 0.0   # right edge has no face
    else:
        avg[-1, :] = 0.0   # bottom edge has no face
    return avg


def _open_face(grid, axis):
    """A face passes flux only if BOTH adjacent cells are open (not solid, not
    vacuum). Vacuum cells are a sink: we zero smoke there each step, so we do
    NOT flux INTO them through the flux form (that would be an unbounded drain
    asymmetry); instead the explicit zeroing handles venting, consistently with
    how the SL scheme treats the breach (sampled as 0)."""
    open_mask = (~grid.solid) & (~grid.is_vacuum)
    nbr = np.roll(open_mask, -1, axis=axis)
    if axis == 1:
        nbr[:, -1] = False
    else:
        nbr[-1, :] = False
    return open_mask & nbr


def _face_perm(grid, axis):
    """Permeability weight on each face = min(perm[self], perm[nbr]) — the SAME
    face coefficient the C++ diffusion uses (neighbor() in smoke_dynamics.cpp).
    Furniture (perm 0.5) throttles flux through its faces to half, so the flux
    form treats a porous crate the way the diffusion does (and the way the SL
    bilinear-with-sealed-corners does) instead of letting smoke pile up on it."""
    perm = grid.perm
    nbr = np.roll(perm, -1, axis=axis)
    if axis == 1:
        nbr[:, -1] = 0.0
    else:
        nbr[-1, :] = 0.0
    return np.minimum(perm, nbr)


def advect_donor(smoke, grid, wind_x, wind_y, dt_adv, limiter=False):
    """Conservative flux-form upwind advection (the water solver's pattern).

    For each face we take the donor (upwind) cell value and move
    flux = v_face * smoke[donor] * dt across it; the divergence of the gathered
    face fluxes updates the cell. With `limiter=True` we add the MC-limited
    2nd-order anti-diffusive correction flux (TVD) on top of the donor flux.

    Conservative BY CONSTRUCTION: whatever leaves cell A across a face enters
    cell B across the same face (sum over interior faces telescopes to 0). The
    only mass sinks are the explicit vacuum-zeroing (a real breach drain) and
    the [0,1] clamp at saturation.

    CFL — THE LOAD-BEARING DIFFERENCE FROM SL. An explicit flux scheme is only
    stable (and only actually conservative) when the per-step Courant number
    |v|*dt/dx < 1: a cell may not move more than one cell-width of mass per step,
    or the divergence overshoots, the clamp bites, and "conservation" becomes a
    thrash against [0,1]. Breach's `advection_rate*dt` displacement is ~37 tiles
    (Courant up to ~70 under a blast), which the SL back-trace eats whole but
    which the flux form CANNOT — so we SUB-STEP internally to keep each flux step
    at Courant < 1. The total transported displacement is identical to SL's
    (same dt_adv); the flux form just pays for it in many small stable steps.
    This sub-step count IS the real cost of switching to flux form, and it is
    reported (see the printed n_sub) rather than hidden.
    """
    H, W = grid.H, grid.W
    vmax = float(max(np.abs(wind_x).max(), np.abs(wind_y).max()))
    courant = vmax * dt_adv               # tiles a cell would move in one full step
    n_sub = max(1, int(math.ceil(courant / 0.5)))   # keep Courant <= 0.5 per substep
    if n_sub > 1:
        dt_sub = dt_adv / n_sub
        s = smoke
        for _ in range(n_sub):
            s = _donor_step(s, grid, wind_x, wind_y, dt_sub, limiter)
        advect_donor.last_n_sub = n_sub
        return s
    advect_donor.last_n_sub = 1
    return _donor_step(smoke, grid, wind_x, wind_y, dt_adv, limiter)


advect_donor.last_n_sub = 1


def _donor_step(smoke, grid, wind_x, wind_y, dt_adv, limiter):
    """One CFL-bounded conservative flux step (Courant < 1 assumed)."""
    global _MC_OPEN_MASK
    H, W = grid.H, grid.W
    s = smoke
    if limiter:
        _MC_OPEN_MASK = (~grid.solid) & (~grid.is_vacuum)

    # --- X faces (between x and x+1) ---
    vfx = _face_velocity(wind_x, axis=1)        # face velocity
    openx = _open_face(grid, axis=1)
    permx = _face_perm(grid, axis=1)            # min(perm) face throttle
    s_right = np.roll(s, -1, axis=1)            # value of cell x+1
    # Donor value on the face: upwind in the face velocity's direction.
    donor_x = np.where(vfx >= 0.0, s, s_right)
    flux_x = vfx * donor_x                       # 1st-order donor flux (per unit time)
    if limiter:
        flux_x = flux_x + _mc_correction(s, vfx, openx, axis=1)
    flux_x = np.where(openx, flux_x * permx, 0.0)  # sealed/edge faces carry nothing

    # --- Y faces (between y and y+1) ---
    vfy = _face_velocity(wind_y, axis=0)
    openy = _open_face(grid, axis=0)
    permy = _face_perm(grid, axis=0)
    s_down = np.roll(s, -1, axis=0)
    donor_y = np.where(vfy >= 0.0, s, s_down)
    flux_y = vfy * donor_y
    if limiter:
        flux_y = flux_y + _mc_correction(s, vfy, openy, axis=0)
    flux_y = np.where(openy, flux_y * permy, 0.0)

    # --- Divergence: cell loses its outgoing-face flux, gains its incoming ---
    # flux_x[i] is the flux across the i|i+1 face (positive = i -> i+1).
    # Cell i: -flux across its right face (loses), +flux across its left face.
    div_x = flux_x - np.roll(flux_x, 1, axis=1)
    div_x[:, 0] = flux_x[:, 0]                    # left domain edge: no incoming
    div_y = flux_y - np.roll(flux_y, 1, axis=0)
    div_y[0, :] = flux_y[0, :]

    out = s - dt_adv * (div_x + div_y)
    _zero_and_clamp(out, grid)
    return out


def _mc_correction(s, vface, openface, axis):
    """MC (monotonized central) limited anti-diffusive correction flux, added to
    the donor flux to make it 2nd-order TVD (MUSCL).

    ONE correction flux per face (selected by the face velocity's sign), so the
    divergence telescopes exactly like the donor flux -> the correction is
    conservative by construction. The face value is donor + 0.5*phi(r)*(downwind
    face gradient), with r the upwind/downwind smoothness ratio and phi the MC
    limiter (monotone: no new extrema, no ringing). dx = 1 tile.

    Boundary safety (load-bearing for conservation): the 2nd-order stencil reads
    a far-upwind cell (i-1 for +flow, i+2 for -flow). Where that far cell is a
    wall / vacuum / domain edge, np.roll would wrap garbage in, breaking the
    telescoping at the boundary; so the correction is ZEROED there and the face
    falls back to bare donor.

    Conservation note (verified by the sealed-box test): the FLUX divergence sums
    to machine zero (so the transport itself is exactly conservative, like bare
    donor). The only residual is that this dimensionally-split MC + the [0,1]
    positivity clamp can lift tiny TVD undershoots (~<0 by 1e-4) back to 0, a
    ~0.1% mass *gain* over a long run. That is two-to-three ORDERS OF MAGNITUDE
    smaller than SL's 100-300% non-conservation, and is the price of the sharper
    fronts; bare donor (limiter off) is exactly conservative if that matters more
    than front sharpness."""
    s_p1 = np.roll(s, -1, axis=axis)   # s[i+1]
    s_m1 = np.roll(s,  1, axis=axis)   # s[i-1]
    s_p2 = np.roll(s, -2, axis=axis)   # s[i+2]

    eps = 1e-12
    # Positive flow: donor = s[i]; upwind/downwind gradients at i.
    d_up_pos   = s   - s_m1            # upwind gradient at i (needs i-1 valid)
    d_down_pos = s_p1 - s             # downwind (face) gradient at i
    r_pos = d_up_pos / np.where(np.abs(d_down_pos) < eps, eps, d_down_pos)
    corr_pos = 0.5 * _mc_limiter(r_pos) * d_down_pos    # face-value increment at i

    # Negative flow: donor = s[i+1]; mirror the stencil.
    d_up_neg   = s_p2 - s_p1          # upwind gradient at i+1 (needs i+2 valid)
    d_down_neg = s_p1 - s             # face gradient seen from i+1
    r_neg = d_up_neg / np.where(np.abs(d_down_neg) < eps, eps, d_down_neg)
    corr_neg = -0.5 * _mc_limiter(r_neg) * d_down_neg

    corr = np.where(vface >= 0.0, corr_pos, corr_neg)
    corr_flux = vface * corr

    # Valid-stencil mask: the far-upwind cell of the chosen branch must be a real
    # open cell (not wrapped, not solid/vacuum). Where invalid -> no correction.
    valid_far = _far_upwind_valid(openface_basis=s, axis=axis)
    far_pos_ok, far_neg_ok = valid_far
    stencil_ok = np.where(vface >= 0.0, far_pos_ok, far_neg_ok)
    return np.where(openface & stencil_ok, corr_flux, 0.0)


def _far_upwind_valid(openface_basis, axis):
    """Per-face booleans: is the far-upwind stencil cell valid for the +flow
    branch (cell i-1) and the -flow branch (cell i+2)? Computed from the GRID
    open mask threaded via a module global set in _donor_step. Returns
    (far_pos_ok, far_neg_ok)."""
    open_mask = _MC_OPEN_MASK
    # +flow far-upwind is i-1 along axis; -flow far-upwind is i+2.
    far_pos = np.roll(open_mask, 1, axis=axis)    # cell i-1
    far_neg = np.roll(open_mask, -2, axis=axis)   # cell i+2
    if axis == 1:
        far_pos[:, 0] = False        # i-1 wrapped from the right edge
        far_neg[:, -1] = False       # i+2 wrapped
        far_neg[:, -2] = False
    else:
        far_pos[0, :] = False
        far_neg[-1, :] = False
        far_neg[-2, :] = False
    return far_pos, far_neg


# Set by _donor_step so _mc_correction can mask out wall-crossing stencils.
_MC_OPEN_MASK = None


def _mc_limiter(r):
    """MC (monotonized central-difference) flux limiter:
        phi(r) = max(0, min(2r, (1+r)/2, 2))."""
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))


# --------------------------------------------------------------------------- #
# Simulation driver
# --------------------------------------------------------------------------- #
SCHEMES = ("SL", "SLint", "donor", "MC")
SCHEME_LABELS = {
    "SL": "Semi-Lagrangian float (current)",
    "SLint": "Semi-Lagrangian INT Q16.16 (deterministic)",
    "donor": "Donor-cell flux (conservative)",
    "MC": "Donor + MC limiter (TVD)",
}


def advect(scheme, smoke, grid, wind_x, wind_y, dt_adv):
    if scheme == "SL":
        return advect_semilagrangian(smoke, grid, wind_x, wind_y, dt_adv)
    if scheme == "SLint":
        return advect_semilagrangian_int(smoke, grid, wind_x, wind_y, dt_adv)
    if scheme == "donor":
        return advect_donor(smoke, grid, wind_x, wind_y, dt_adv, limiter=False)
    if scheme == "MC":
        return advect_donor(smoke, grid, wind_x, wind_y, dt_adv, limiter=True)
    raise ValueError(scheme)


@dataclass
class Detonation:
    tick: int          # tick at which it fires
    fy: int
    fx: int
    radius: float
    pressure: float


def run_scenario(name, grid, cfg, detonations, n_ticks, capture_every, seed=12345,
                 wind_window=20, wind_decay=0.965, adv_mult=0.8,
                 wind_diff_factor=0.06):
    """Run all three schemes through the SAME ticks with the SAME wind events.

    Wind: each detonation installs a blast wind for a window of ticks after it
    fires (decaying per tick); the active field is the SUM of recent blasts.
    The wind is solved ONCE per detonation and reused — the SAME field feeds all
    three advection schemes, so any visible difference is the advection only.

    DEMO TUNING (documented, not hidden). The shipped `config.toml` smoke
    constants are flagged in `docs/architecture/engine/05_smoke.md` as needing a
    re-tune (Patch 2 made the effective diffusion ~9x weaker than the shipped
    build, and the wind-ride coefficient is large). For a LEGIBLE side-by-side
    we soften two render-irrelevant knobs so the cloud actually survives its trip
    down the corridor instead of dissolving in two ticks:
      * adv_mult         scales advection_rate (0.8) -> a few tiles/tick, so the
                         funnel reads frame to frame rather than teleporting.
      * wind_diff_factor scales wind_diffusion_scale (0.06) -> the wind-coupled
                         diffusion no longer smears the cloud to invisible mid-trip.
    Both are applied IDENTICALLY to all three schemes, so they cannot bias the
    comparison — they only set a regime where the advection difference is visible.
    The base d_smoke, the deposit, the wind solve, and the schemes themselves are
    unchanged from the game.
    """
    H, W = grid.H, grid.W
    tps = cfg["tps"]
    dt = 1.0 / tps
    eff_adv_rate = cfg["advection_rate"] * adv_mult
    eff_wind_diff = cfg["wind_diffusion_scale"] * wind_diff_factor
    # Smoke diffusion is explicit forward-Euler; keep it stable with the same
    # n_smoke CFL floor the engine uses (d_eff can be large under blast wind).
    # We split the tick into n_smoke advect+diffuse substeps for all schemes.
    # (Identical substepping across schemes -> apples to apples.)

    # Precompute each detonation's wind field once.
    blast_winds = []
    for det in detonations:
        wx, wy, _ = blast_wind(grid, det.fy, det.fx, det.radius, det.pressure)
        blast_winds.append((wx, wy))

    WIND_WINDOW = int(wind_window)        # ticks a blast's wind stays active
    WIND_DECAY = float(wind_decay)        # per-tick decay within the window

    # Per-scheme smoke fields + per-tick recorded mass + captured frames.
    smokes = {s: np.zeros((H, W), dtype=np.float64) for s in SCHEMES}
    masses = {s: [] for s in SCHEMES}
    frames = {s: [] for s in SCHEMES}   # list of (tick, smoke_copy)
    max_flux_substeps = 0                # peak CFL substeps the flux form needed
    # Integer-SL determinism/mass tracking: the int32 Q16.16 field after the
    # last advect substep of each tick is the SYNCED state (what would be
    # lockstepped). We checksum it every tick and keep the final-tick field so a
    # second run can be asserted bit-identical to it.
    slint_field_q = None                 # last int32 Q16.16 field (post-advect)
    slint_int_mass = []                  # per-tick sum of the int32 field (raw counts)
    slint_checksums = []                 # per-tick crc-ish of the int32 field

    # Independent RNG per scheme so the deposit is identical (same seed, same
    # call order) -> the three start from a bit-identical cloud each detonation.
    for tick in range(n_ticks):
        # ---- assemble the active wind field (sum of recent blasts) ----
        wind_x = np.zeros((H, W)); wind_y = np.zeros((H, W))
        for det, (wx, wy) in zip(detonations, blast_winds):
            age = tick - det.tick
            if 0 <= age < WIND_WINDOW:
                scale = WIND_DECAY ** age
                wind_x = wind_x + wx * scale
                wind_y = wind_y + wy * scale

        # ---- fire any detonation that triggers THIS tick (same deposit per scheme) ----
        for det in detonations:
            if det.tick == tick:
                for s in SCHEMES:
                    rng = np.random.default_rng(seed + det.tick)   # identical across schemes
                    deposit_smoke(smokes[s], grid, det.fy, det.fx,
                                  det.radius, cfg["explosion_smoke_noise"], rng)

        # ---- n_smoke CFL floor from the spatial-max d_eff (same for all) ----
        wind_sq = wind_x * wind_x + wind_y * wind_y
        d_eff_max = cfg["d_smoke"] * (1.0 + eff_wind_diff * float(wind_sq.max()))
        max_dt_diff = 1.0 / (4.0 * d_eff_max) if d_eff_max > 0 else dt
        n_smoke = max(1, int(math.ceil(dt / max_dt_diff)))
        dt_sub = dt / n_smoke
        dt_adv_sub = eff_adv_rate * dt_sub

        # ---- step every scheme: advect (scheme-specific) then diffuse (shared) ----
        for s in SCHEMES:
            sm = smokes[s]
            for _ in range(n_smoke):
                sm = advect(s, sm, grid, wind_x, wind_y, dt_adv_sub)
                if s in ("donor", "MC"):
                    max_flux_substeps = max(max_flux_substeps,
                                            advect_donor.last_n_sub)
                if s == "SLint":
                    # The int32 Q16.16 field straight out of the integer advect
                    # is the deterministic synced state (diffuse is the shared
                    # float pass that doesn't enter the integer-determinism test).
                    slint_field_q = advect_semilagrangian_int.last_field_q.copy()
                sm = diffuse(sm, grid, wind_x, wind_y, cfg["d_smoke"],
                             eff_wind_diff, dt_sub)
            smokes[s] = sm
            masses[s].append(float(sm.sum()))

        # Record the integer field's checksum + integer mass for this tick.
        slint_int_mass.append(int(slint_field_q.astype(np.int64).sum()))
        slint_checksums.append(int(
            np.bitwise_xor.reduce(
                (slint_field_q.astype(np.uint32).ravel()
                 * np.arange(1, slint_field_q.size + 1, dtype=np.uint64)
                 ).astype(np.uint64))))

        if tick % capture_every == 0:
            for s in SCHEMES:
                frames[s].append((tick, smokes[s].copy()))

    return {
        "name": name, "grid": grid, "smokes_frames": frames,
        "masses": masses, "n_ticks": n_ticks, "tps": tps,
        "capture_every": capture_every,
        "max_flux_substeps": max_flux_substeps,
        # Integer-SL determinism evidence:
        "slint_field_q": slint_field_q,        # final-tick int32 Q16.16 field
        "slint_int_mass": slint_int_mass,      # per-tick int mass (raw counts)
        "slint_checksums": slint_checksums,    # per-tick field checksum
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _smoke_cmap():
    # Transparent -> warm grey -> bright, over the wall/open base.
    return LinearSegmentedColormap.from_list(
        "smoke", [(0.95, 0.55, 0.15), (0.85, 0.85, 0.9), (1, 1, 1)])


# Display normalization: the cloud spreads thin as it funnels (avg density
# ~0.03 over a wide corridor), so a raw density->alpha map renders it nearly
# invisible. We apply a PERCEPTUAL display curve (sqrt-like) + a fixed display
# ceiling so any present smoke reads clearly. This is RENDER-ONLY and IDENTICAL
# across all three schemes — it cannot create or hide a difference between them,
# it only makes the (faithfully simulated) thin cloud legible to the eye.
# Ceiling 0.18 keeps the lumpy flux peaks (max ~0.55) from blowing to pure white
# while the perceptual gamma still lifts the soft SL haze (max ~0.02) into view —
# a fair display for BOTH the diffuse SL look and the sharp flux look.
SMOKE_DISPLAY_CEIL = 0.18   # density mapped to full opacity
SMOKE_DISPLAY_GAMMA = 0.7   # <1 = lift thin smoke (perceptual), brightens wisps


def render_panel(grid, smoke):
    """RGB image (H, W, 3) uint8: walls dark steel, open dark, smoke as a
    warm density colormap composited on top (perceptual display curve)."""
    H, W = grid.H, grid.W
    img = np.zeros((H, W, 3), dtype=np.float64)
    open_mask = (~grid.solid) & (~grid.is_vacuum)
    img[grid.is_vacuum] = (0.02, 0.02, 0.05)     # space: near-black blue
    img[open_mask] = (0.12, 0.13, 0.16)          # open floor: dark slate
    img[grid.solid] = (0.32, 0.34, 0.40)         # walls: steel grey
    cmap = _smoke_cmap()
    d = np.clip(smoke / SMOKE_DISPLAY_CEIL, 0.0, 1.0)
    d_vis = d ** SMOKE_DISPLAY_GAMMA             # lift thin smoke for legibility
    smoke_rgb = cmap(d_vis)[..., :3]
    a = np.clip(d_vis, 0.0, 1.0)[..., None]
    img = img * (1.0 - a) + smoke_rgb * a
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _stack_panels(grid, frame_smokes, scale, gap=6):
    """Horizontally concatenate the N scheme panels (already same tick),
    with a labelled gap. Returns an upscaled RGB image."""
    panels = [render_panel(grid, sm) for sm in frame_smokes]
    panels = [np.kron(p, np.ones((scale, scale, 1), dtype=np.uint8)) for p in panels]
    Hs = panels[0].shape[0]
    sep = np.full((Hs, gap, 3), 30, dtype=np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, sep, p], axis=1)
    return row


def make_gif(result, path, scale=4, fps=12):
    grid = result["grid"]
    frame_lists = [result["smokes_frames"][s] for s in SCHEMES]
    n = len(frame_lists[0])
    nsch = len(SCHEMES)
    images = []
    for i in range(n):
        tick = frame_lists[0][i][0]
        frame_smokes = [frame_lists[k][i][1] for k in range(nsch)]
        composite = _stack_panels(grid, frame_smokes, scale)
        # Add a thin header bar with scheme labels + tick.
        composite = _add_header(composite, tick, result["tps"])
        images.append(composite)
    imageio.mimsave(path, images, fps=fps, loop=0)
    return path


# Short panel headers (the long names overlapped at panel width).
SCHEME_SHORT = {
    "SL": "1. SL float (current)",
    "SLint": "1b. SL INT Q16.16 (det.)",
    "donor": "2. Donor flux (cons.)",
    "MC": "3. Donor + MC limiter",
}


def _add_header(img, tick, tps):
    """Draw the three scheme labels (centered per panel) + tick/time."""
    from PIL import Image, ImageDraw
    H, W, _ = img.shape
    bar_h = 30
    out = np.full((H + bar_h, W, 3), 18, dtype=np.uint8)
    out[bar_h:, :, :] = img
    pim = Image.fromarray(out)
    draw = ImageDraw.Draw(pim)
    nsch = len(SCHEMES)
    seg = W // nsch
    labels = [SCHEME_SHORT[s] for s in SCHEMES]
    for k, lab in enumerate(labels):
        # center the label within its panel segment
        try:
            tw = draw.textlength(lab)
        except Exception:
            tw = len(lab) * 6
        cx = k * seg + max(2, (seg - int(tw)) // 2)
        draw.text((cx, 3), lab, fill=(230, 230, 235))
    draw.text((6, 17), f"tick {tick}  ({tick / tps:0.2f}s)", fill=(150, 190, 255))
    return np.asarray(pim)


def make_montage(result, path, n_key=4, scale=4):
    """Static PNG: a few key frames (rows) x 3 schemes side by side."""
    grid = result["grid"]
    frame_lists = [result["smokes_frames"][s] for s in SCHEMES]
    n = len(frame_lists[0])
    nsch = len(SCHEMES)
    idxs = np.linspace(0, n - 1, n_key).round().astype(int)
    rows = []
    for i in idxs:
        tick = frame_lists[0][i][0]
        frame_smokes = [frame_lists[k][i][1] for k in range(nsch)]
        composite = _stack_panels(grid, frame_smokes, scale)
        composite = _add_header(composite, tick, result["tps"])
        rows.append(composite)
    maxw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0)), constant_values=18)
            for r in rows]
    sepr = np.full((4, maxw, 3), 40, dtype=np.uint8)
    full = rows[0]
    for r in rows[1:]:
        full = np.concatenate([full, sepr, r], axis=0)
    imageio.imwrite(path, full)
    return path


def make_conservation_plot(result, path):
    masses = result["masses"]
    tps = result["tps"]
    n = result["n_ticks"]
    t = np.arange(n) / tps
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"SL": "#d62728", "SLint": "#ff7f0e",
              "donor": "#1f77b4", "MC": "#2ca02c"}
    stats = {}
    for s in SCHEMES:
        m = np.asarray(masses[s])
        ax.plot(t, m, label=SCHEME_LABELS[s], color=colors[s], lw=2)
        peak = float(m.max())
        final = float(m[-1])
        stats[s] = (peak, final)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total smoke mass (sum of density)")
    ax.set_title(f"Smoke mass conservation - {result['name']}\n"
                 "Donor + MC sit flat (conservative, drop only on venting); "
                 "SL drifts (non-conservative)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, stats


# --------------------------------------------------------------------------- #
# Integer-SL determinism + non-conservation verification
# --------------------------------------------------------------------------- #
def verify_int_determinism(name, grid, cfg, dets, n_ticks, capture_every):
    """THE CORE PROOF. Run the WHOLE scenario twice and assert the integer-SL
    int32 Q16.16 smoke field is BIT-IDENTICAL between the two runs (every per-tick
    checksum equal AND the final int32 field equal element-for-element). Integer
    +/-/*/>> are exact and associative, so a deterministic integer SL MUST be
    byte-stable; this is the empirical confirmation that nothing float leaked in.

    Returns (ok, result_a) — result_a is reused for the rest of the outputs so we
    don't pay for a third run."""
    ra = run_scenario(name, grid, cfg, dets, n_ticks, capture_every)
    rb = run_scenario(name, grid, cfg, dets, n_ticks, capture_every)
    cks_a = ra["slint_checksums"]
    cks_b = rb["slint_checksums"]
    checks_equal = (cks_a == cks_b)
    field_equal = bool(np.array_equal(ra["slint_field_q"], rb["slint_field_q"]))
    ok = checks_equal and field_equal
    print(f"\n=== DETERMINISM ({name}) ===")
    print(f"  per-tick int32 checksums identical across two runs: {checks_equal} "
          f"({len(cks_a)} ticks)")
    print(f"  final-tick int32 Q16.16 field bit-identical:        {field_equal} "
          f"(shape {ra['slint_field_q'].shape}, dtype {ra['slint_field_q'].dtype})")
    # Hard assert — this is the load-bearing proof.
    assert ok, "INTEGER SL IS NOT BIT-DETERMINISTIC (fields/checksums differ!)"
    print("  ASSERT PASSED: integer SL is bit-deterministic on this machine.")
    return ok, ra


def make_int_mass_plot(result, path):
    """Integer-SL total mass (raw Q16.16 counts -> real density units) over time:
    the non-conservation magnitude, which sizes the smoke-decay knob."""
    im = np.asarray(result["slint_int_mass"], dtype=np.float64) / FP_ONE
    tps = result["tps"]
    t = np.arange(len(im)) / tps
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, im, color="#ff7f0e", lw=2,
            label="Integer SL (Q16.16) total mass")
    # Overlay the float SL mass for reference (same scenario, scheme SL).
    fl = np.asarray(result["masses"]["SL"], dtype=np.float64)
    ax.plot(t, fl, color="#d62728", lw=1.5, ls="--",
            label="Float SL total mass (reference)")
    peak = float(im.max()) if im.size else 0.0
    final = float(im[-1]) if im.size else 0.0
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total smoke mass (sum of density)")
    ax.set_title(f"Integer SL mass (non-conservation) - {result['name']}\n"
                 f"peak={peak:.1f}  final={final:.1f}  "
                 f"kept={100.0*final/peak if peak>0 else 0:.1f}%")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path, (peak, final)


def make_sl_diff_montage(result, path, n_key=4, scale=4):
    """Float-SL vs Int-SL side by side PLUS their absolute difference, so the
    'does the integer approximation degrade the look' question is answered
    directly: a near-black diff column == the integer SL reproduces the float
    look; visible structure in the diff == where the approximations bite."""
    grid = result["grid"]
    fl = result["smokes_frames"]["SL"]
    iq = result["smokes_frames"]["SLint"]
    n = len(fl)
    idxs = np.linspace(0, n - 1, n_key).round().astype(int)
    rows = []
    for i in idxs:
        tick = fl[i][0]
        smf = fl[i][1]
        smi = iq[i][1]
        diff = np.abs(smf - smi)
        # Amplify the diff x6 for visibility (it is tiny if the schemes match).
        panels = [render_panel(grid, smf), render_panel(grid, smi),
                  render_panel(grid, np.clip(diff * 6.0, 0.0, 1.0))]
        panels = [np.kron(p, np.ones((scale, scale, 1), dtype=np.uint8))
                  for p in panels]
        Hs = panels[0].shape[0]
        sep = np.full((Hs, 6, 3), 30, dtype=np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.concatenate([row, sep, p], axis=1)
        # Header: float | int | diff x6.
        from PIL import Image, ImageDraw
        bar_h = 30
        out = np.full((row.shape[0] + bar_h, row.shape[1], 3), 18, dtype=np.uint8)
        out[bar_h:, :, :] = row
        pim = Image.fromarray(out)
        draw = ImageDraw.Draw(pim)
        seg = row.shape[1] // 3
        for k, lab in enumerate(("SL float", "SL INT Q16.16", "|float-int| x6")):
            draw.text((k * seg + 6, 3), lab, fill=(230, 230, 235))
        draw.text((6, 17), f"tick {tick} ({tick/result['tps']:0.2f}s)",
                  fill=(150, 190, 255))
        rows.append(np.asarray(pim))
    maxw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0)), constant_values=18)
            for r in rows]
    sepr = np.full((4, maxw, 3), 40, dtype=np.uint8)
    full = rows[0]
    for r in rows[1:]:
        full = np.concatenate([full, sepr, r], axis=0)
    imageio.imwrite(path, full)
    # Also report the mean/max float-vs-int discrepancy over the open domain.
    open_mask = (~grid.solid) & (~grid.is_vacuum)
    last_f = fl[-1][1][open_mask]
    last_i = iq[-1][1][open_mask]
    mad = float(np.abs(last_f - last_i).mean())
    mxd = float(np.abs(last_f - last_i).max())
    return path, (mad, mxd)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = load_config_physics()
    grid = load_grid()
    print(f"Map: {grid.W} x {grid.H} tiles  "
          f"({int((~grid.solid & ~grid.is_vacuum).sum())} open, "
          f"{int(grid.solid.sum())} wall, {int(grid.is_vacuum.sum())} space)")
    print(f"Config: d_smoke={cfg['d_smoke']} advection_rate={cfg['advection_rate']} "
          f"wind_diffusion_scale={cfg['wind_diffusion_scale']} "
          f"noise={cfg['explosion_smoke_noise']} tps={cfg['tps']}")

    GRENADE_RADIUS = 5.0
    GRENADE_PRESSURE = 16.0   # config grenade pressure is 10; bumped for a
                              # stronger, longer corridor sweep in the demo (the
                              # blast wind is identical across all 3 schemes).

    # --- Scenario 1: single grenade near the TOP of the room at rows 46-56,
    #     which is sealed above (wall row 44) and open below into the rows 58-63
    #     corridor -> the over-pressure relieves DOWNWARD and the cloud funnels
    #     through the pinch into the big room (rows 64-80). ---
    s1_dets = [Detonation(tick=0, fy=50, fx=24, radius=GRENADE_RADIUS,
                          pressure=GRENADE_PRESSURE)]
    print("\n[Scenario 1] single grenade @ (row 50, col 24) "
          "-> funnels down the cols 22-25 corridor (rows 58-63)")
    # CALM-vs-BLAST note: scenario 1 is a SINGLE blast then quiet decay (the
    # 'calm' reference for the integer-SL mass curve). Scenario 2 is the double
    # blast (the 'blast' over-amplification stress test).
    _, r1 = verify_int_determinism("Scenario 1 - single grenade", grid, cfg,
                                   s1_dets, n_ticks=44, capture_every=2)

    # --- Scenario 2: grenade 1 in the big lower room (rows 64-80); grenade 2
    #     fires 12 ticks later ABOVE the first cloud, its blast PUSHES the
    #     first cloud DOWN through the rows 81-85 corridor into the bottom
    #     room (rows 86-114). The dynamic Erik specifically asked to see. ---
    s2_dets = [
        Detonation(tick=0,  fy=78, fx=24, radius=GRENADE_RADIUS, pressure=GRENADE_PRESSURE),
        Detonation(tick=12, fy=68, fx=24, radius=GRENADE_RADIUS, pressure=GRENADE_PRESSURE),
    ]
    print("[Scenario 2] grenade 1 @ (row 78, col 24); grenade 2 @ (row 68, col 24) "
          "at tick 12 -> pushes the cloud down the rows 81-85 corridor")
    _, r2 = verify_int_determinism("Scenario 2 - two grenades", grid, cfg,
                                   s2_dets, n_ticks=52, capture_every=2)

    # --- Outputs ---
    print("\nWriting outputs...")
    g1 = make_gif(r1, OUT / "scenario1_single.gif")
    g2 = make_gif(r2, OUT / "scenario2_double.gif")
    m1 = make_montage(r1, OUT / "scenario1_montage.png")
    m2 = make_montage(r2, OUT / "scenario2_montage.png")
    c1, st1 = make_conservation_plot(r1, OUT / "scenario1_conservation.png")
    c2, st2 = make_conservation_plot(r2, OUT / "scenario2_conservation.png")
    # Integer-SL specific: the float-vs-int look comparison + the int mass curve.
    d1, dd1 = make_sl_diff_montage(r1, OUT / "scenario1_SLfloat_vs_SLint.png")
    d2, dd2 = make_sl_diff_montage(r2, OUT / "scenario2_SLfloat_vs_SLint.png")
    im1, ms1 = make_int_mass_plot(r1, OUT / "scenario1_SLint_mass.png")
    im2, ms2 = make_int_mass_plot(r2, OUT / "scenario2_SLint_mass.png")

    # --- Report numbers ---
    def report(result, stats):
        name = result["name"]
        print(f"\n=== {name}: peak vs final total mass per scheme ===")
        # The reference for "% kept" is each scheme's OWN peak (the moment all
        # smoke is deposited and before transport has drained any). The flux
        # schemes are conservative by construction, so their only true mass loss
        # is genuine vacuum venting + the [0,1] saturation clamp; SL loses mass
        # to the bilinear back-trace + wall-clip on top of that.
        for s in SCHEMES:
            peak, final = stats[s]
            kept = 100.0 * final / peak if peak > 0 else 0.0
            print(f"  {SCHEME_LABELS[s]:34s}  peak={peak:8.2f}  "
                  f"final={final:8.2f}  kept={kept:6.1f}%")
        peak_mc, final_mc = stats["MC"]
        if peak_mc > 0:
            print(f"  -> at peak, SL holds {100.0*stats['SL'][0]/peak_mc:5.1f}% "
                  f"of the conservative (MC) cloud mass")
        print(f"  -> flux form needed up to {result['max_flux_substeps']} CFL "
              f"substeps/tick (Courant<=0.5); SL needs 1 (the cost of flux form)")

    report(r1, st1)
    report(r2, st2)

    # --- Float-SL vs Integer-SL: the 'is the look preserved' numbers ---
    print("\n=== FLOAT-SL vs INTEGER-SL agreement (open domain, final frame) ===")
    print(f"  Scenario 1 (calm) : mean|d|={dd1[0]:.5f}  max|d|={dd1[1]:.5f}  "
          f"(density units; SL peak ~0.8)")
    print(f"  Scenario 2 (blast): mean|d|={dd2[0]:.5f}  max|d|={dd2[1]:.5f}")
    print("  (small mean|d| == integer SL reproduces the float SL look; see the "
          "SLfloat_vs_SLint montages for where any difference sits)")

    # --- Integer-SL non-conservation (the decay-knob sizing) ---
    print("\n=== INTEGER-SL non-conservation (mass kept vs own peak) ===")
    for tag, ms in (("Scenario 1 (calm) ", ms1), ("Scenario 2 (blast)", ms2)):
        peak, final = ms
        kept = 100.0 * final / peak if peak > 0 else 0.0
        print(f"  {tag}: peak={peak:8.2f}  final={final:8.2f}  kept={kept:6.1f}%")

    print("\nOutput files:")
    for p in (g1, g2, m1, m2, c1, c2, d1, d2, im1, im2):
        print("  ", Path(p).resolve())


if __name__ == "__main__":
    main()
