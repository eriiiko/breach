"""The smoke BILLOW noise baker (graphics arc G1, docs/graphics_direction_
2026-10-18.md §2.1, §8.2) -- pyray-free, numpy only, deterministic.

Why a new noise. The B2 detail pass erodes the smoke with fBm, a homogeneous
Gaussian-ish texture: it carves alpha but has no FORM. Smoke seen from above
reads as smoke because of CONVEX LUMPS -- billows with a solid core and a
fairly hard boundary. That shape is cellular noise turned inside out:
inverted Worley (one dome per feature point), several octaves of it, warped
by a low-frequency fBm so the cells do not sit on a visible lattice, and
remapped by a Perlin fBm so whole clusters of lumps swell and shrink
together (the Nubis "Perlin-Worley" recipe). The relief shader
(``shaders/gas_relief.fs``) reads it as the billow field ``B`` that pushes
the coverage edge in and out and gives the relief height its bulging top.

Every field here is a function of CONTINUOUS coordinates in texture units
that is EXACTLY periodic with period 1 on every axis (integer lattice
periods, wrapped feature points, a periodic warp), so a bake on any grid
tiles seamlessly; ``tests/test_noise_bake.py`` holds that.

What this module bakes:
  * :func:`bake_billow_rgba` -- the 2D texture the top-down relief samples
    (REPEAT + BILINEAR): R = the billow field B, G/B = a decorrelated
    periodic fBm pair (the shader's domain warp), A = one octave of high-
    frequency inverted Worley (fine detail for the edge).
  * :func:`bake_billow_3d` -- the same billow recipe on a periodic cube, the
    3D variant for the tilted cinematic integrator (§2.4, the ``#90``
    raymarcher, G6). Nothing in G1 draws it yet.

Fixed seeds (``numpy.random.default_rng``, PCG64): byte-identical bakes on
every run and machine.

Credit (repo rule -- cite what a file implements):
  - Steven Worley, "A Cellular Texture Basis Function", SIGGRAPH 1996 --
    cellular (F1) noise.
  - Ken Perlin, "Improving Noise", SIGGRAPH 2002 -- gradient noise with the
    quintic fade.
  - Andrew Schneider & Nathan Vos, "The Real-Time Volumetric Cloudscapes of
    Horizon Zero Dawn", SIGGRAPH 2015 Advances in Real-Time Rendering --
    Perlin-Worley: inverted Worley octaves remapped by Perlin fBm.
  - Inigo Quilez, "Domain Warping" (iquilezles.org/articles/warp) -- the
    fBm warp of the sampling coordinates.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Fixed seeds: each field draws from its own stream, so changing one octave
# count never reshuffles another field.
BILLOW_SEED = 0x6B1110        # "G1 billo(w)"
_WARP_SEED_OFFSET = 101
_DETAIL_SEED_OFFSET = 202
_PERLIN_SEED_OFFSET = 303

# Coarsest Worley lattice: cells per texture period. 4 cells per period means
# the shader's billow scale dial (cell size in tiles) sets a texture period of
# 4 x that dial.
BASE_CELLS = 4
# Octave weights of the inverted-Worley fBm (Schneider & Vos's 0.625 / 0.25 /
# 0.125, normalised): the big lumps dominate, the small ones crinkle them.
_WORLEY_WEIGHTS = (0.625, 0.25, 0.125)
# How deep the Perlin clustering erodes the domes (billow_field step 4).
_EROSION_K = 0.5
# Smooth-min width of the billow cells' creases, in cells (worley_f1).
_CREASE_SMOOTH = 0.12


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin's quintic fade 6t^5 - 15t^4 + 10t^3 (C2 across lattice lines)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ---------------------------------------------------------------------------
# Periodic gradient (Perlin) noise, any dimension 2 or 3
# ---------------------------------------------------------------------------

def _unit_gradients(rng: np.random.Generator, shape: Tuple[int, ...]
                    ) -> np.ndarray:
    """Random unit vectors on a lattice of ``shape`` (last axis = dim)."""
    dim = len(shape)
    g = rng.normal(size=shape + (dim,))
    g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    return g


def perlin(coords: np.ndarray, period: int, rng: np.random.Generator
           ) -> np.ndarray:
    """Periodic gradient noise at ``coords`` (..., dim) in texture units.

    The lattice has ``period`` cells per unit on every axis and its
    gradients wrap modulo ``period``, so the field satisfies
    ``f(x + 1) == f(x)`` on every axis. Returns roughly [-0.7, 0.7]."""
    coords = np.asarray(coords, dtype=np.float64)
    dim = coords.shape[-1]
    grads = _unit_gradients(rng, (period,) * dim)
    p = coords * period
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    w = _fade(f)
    out = np.zeros(coords.shape[:-1], dtype=np.float64)
    for corner in range(1 << dim):
        offs = np.array([(corner >> a) & 1 for a in range(dim)], dtype=np.int64)
        idx = tuple(((i0[..., a] + offs[a]) % period) for a in range(dim))
        g = grads[idx]                                   # (..., dim)
        d = f - offs                                     # (..., dim)
        dot = np.sum(g * d, axis=-1)
        weight = np.ones_like(dot)
        for a in range(dim):
            weight = weight * (w[..., a] if offs[a] else (1.0 - w[..., a]))
        out += weight * dot
    return out


def perlin_fbm(coords: np.ndarray, base_period: int, octaves: int,
               rng: np.random.Generator, persistence: float = 0.5
               ) -> np.ndarray:
    """Periodic Perlin fBm, normalised by the amplitude sum (~[-0.7, 0.7])."""
    acc = np.zeros(np.asarray(coords).shape[:-1], dtype=np.float64)
    amp, amp_sum = 1.0, 0.0
    for o in range(int(octaves)):
        acc += amp * perlin(coords, base_period * (2 ** o), rng)
        amp_sum += amp
        amp *= persistence
    return acc / max(amp_sum, 1e-12)


# ---------------------------------------------------------------------------
# Periodic cellular (Worley F1) noise, inverted into domes
# ---------------------------------------------------------------------------

def worley_f1(coords: np.ndarray, cells: int, rng: np.random.Generator,
              smooth: float = 0.0) -> np.ndarray:
    """Periodic Worley F1 at ``coords`` (..., dim): the distance to the
    nearest feature point, in CELL units (one jittered point per cell; the
    point table wraps modulo ``cells``, so ``f(x + 1) == f(x)``).

    ``smooth > 0`` takes a SMOOTH minimum, ``-k log(sum exp(-d / k))``
    (Quilez's smooth Voronoi): where two cells meet, the hard min has a
    crease -- under a low lamp each one drew a dark line between billows
    ("brain folds", G1 step 6) -- and the smooth min rounds it over ``k``
    cells."""
    coords = np.asarray(coords, dtype=np.float64)
    dim = coords.shape[-1]
    pts = rng.random(size=(cells,) * dim + (dim,))       # in-cell offsets
    p = coords * cells
    i0 = np.floor(p).astype(np.int64)
    best = np.full(coords.shape[:-1], np.inf)
    acc = np.zeros(coords.shape[:-1])
    rng_offs = np.array(np.meshgrid(*([[-1, 0, 1]] * dim), indexing="ij")
                        ).reshape(dim, -1).T
    for offs in rng_offs:
        cell = i0 + offs                                  # (..., dim)
        idx = tuple(cell[..., a] % cells for a in range(dim))
        feat = cell + pts[idx]                            # (..., dim)
        d2 = np.sum((feat - p) ** 2, axis=-1)
        if smooth > 0.0:
            acc += np.exp(-np.sqrt(d2) / smooth)
        else:
            np.minimum(best, d2, out=best)
    if smooth > 0.0:
        return np.maximum(-smooth * np.log(np.maximum(acc, 1e-300)), 0.0)
    return np.sqrt(best)


def worley_domes(coords: np.ndarray, base_cells: int, octaves: int,
                 rng: np.random.Generator, smooth: float = 0.0
                 ) -> np.ndarray:
    """Inverted Worley as DOMES, octave-summed, in [0, 1].

    Each octave is ``sqrt(1 - min(F1, 1)^2)``: a hemisphere over every
    feature point (convex, flat-topped lumps with a steep flank -- the
    billow), rather than the cone ``1 - F1`` gives. Octaves double the cell
    count with the Schneider & Vos weights."""
    weights = _WORLEY_WEIGHTS[:max(1, int(octaves))]
    acc = np.zeros(np.asarray(coords).shape[:-1], dtype=np.float64)
    for o, wgt in enumerate(weights):
        f1 = np.minimum(worley_f1(coords, base_cells * (2 ** o), rng,
                                  smooth), 1.0)
        acc += wgt * np.sqrt(np.maximum(1.0 - f1 * f1, 0.0))
    return acc / sum(weights)


def _remap(v, lo, hi, new_lo, new_hi):
    return new_lo + (v - lo) * (new_hi - new_lo) / np.maximum(hi - lo, 1e-9)


def billow_field(coords: np.ndarray, *, seed: int = BILLOW_SEED,
                 base_cells: int = BASE_CELLS, octaves: int = 3,
                 warp: float = 0.06) -> np.ndarray:
    """THE billow field B at ``coords`` (..., 2 or 3) in texture units,
    exactly periodic with period 1 on every axis; returns [0, 1].

    Recipe (Perlin-Worley, after Schneider & Vos 2015):
      1. warp the coordinates by a periodic two-octave Perlin fBm of
         amplitude ``warp`` (texture units) -- the cells lose their lattice;
      2. ``W`` = inverted-Worley domes over ``octaves`` octaves -- the
         lumps;
      3. ``P`` = a Perlin fBm in [0, 1] on the coarse cell lattice;
      4. ``B = remap(W, k (1 - P), 1, 0, 1)`` with ``k = 1/2``: the domes,
         ERODED where Perlin is low, so the lumps cluster into larger
         billows separated by darker lanes (Schneider & Vos remap Perlin by
         Worley; here the Worley lumps are the form and Perlin the
         clustering, which is what a top-down view of billows needs -- by
         eye on the bake, G1).
    Returns the raw remap clipped to [0, 1]; a bake renormalises it."""
    coords = np.asarray(coords, dtype=np.float64)
    dim = coords.shape[-1]
    rng_w = np.random.default_rng(int(seed) + _WARP_SEED_OFFSET)
    warped = coords.copy()
    for a in range(dim):
        warped[..., a] = coords[..., a] + warp * perlin_fbm(
            coords, base_cells, 2, rng_w)
    rng_c = np.random.default_rng(int(seed))
    w = worley_domes(warped, base_cells, octaves, rng_c, _CREASE_SMOOTH)
    rng_p = np.random.default_rng(int(seed) + _PERLIN_SEED_OFFSET)
    p01 = np.clip(0.5 + perlin_fbm(coords, base_cells, 3, rng_p) * 1.4,
                  0.0, 1.0)
    b = _remap(w, _EROSION_K * (1.0 - p01), 1.0, 0.0, 1.0)
    return np.clip(b, 0.0, 1.0)


def _grid_coords_2d(size: int) -> np.ndarray:
    """(size, size, 2) texel coordinates (x, y) in texture units, row = y."""
    ax = np.arange(size, dtype=np.float64) / size
    y, x = np.meshgrid(ax, ax, indexing="ij")
    return np.stack([x, y], axis=-1)


def _to_u8(v: np.ndarray) -> np.ndarray:
    """Normalise over the realised 0.5 / 99.5 percentiles (a few outlier
    texels must not squeeze the histogram) and quantise to uint8."""
    lo, hi = (float(x) for x in np.percentile(v, (0.5, 99.5)))
    if hi > lo:
        v = (v - lo) / (hi - lo)
    return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)


def bake_billow_rgba(size: int = 256, *, seed: int = BILLOW_SEED,
                     base_cells: int = BASE_CELLS, octaves: int = 3,
                     warp: float = 0.06) -> np.ndarray:
    """Bake the 2D billow texture -> (size, size, 4) uint8, row = y.

    R = the billow field B (:func:`billow_field`), normalised to [0, 255].
    G, B = a decorrelated periodic Perlin fBm pair, the shader's domain warp
           (0.5 = no displacement).
    A = one octave of inverted-Worley domes at 4x the base cell count -- the
        fine edge detail.
    Upload REPEAT + BILINEAR: the texture tiles by construction. ``size``
    should be a multiple of ``base_cells * 2**(octaves+1)`` so every lattice
    line lands on a texel (not required for tiling, only for crispness)."""
    xy = _grid_coords_2d(int(size))            # (..., 2) = (x, y), row = y
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., 0] = _to_u8(billow_field(xy, seed=seed, base_cells=base_cells,
                                      octaves=octaves, warp=warp))
    rng_g = np.random.default_rng(int(seed) + _WARP_SEED_OFFSET + 7)
    out[..., 1] = _to_u8(perlin_fbm(xy, 2, 3, rng_g))
    out[..., 2] = _to_u8(perlin_fbm(xy, 2, 3, rng_g))
    rng_d = np.random.default_rng(int(seed) + _DETAIL_SEED_OFFSET)
    out[..., 3] = _to_u8(worley_domes(xy, base_cells * 4, 1, rng_d))
    return out


def bake_billow_3d(size: int = 64, *, seed: int = BILLOW_SEED,
                   base_cells: int = BASE_CELLS, octaves: int = 3,
                   warp: float = 0.06) -> np.ndarray:
    """Bake the 3D billow variant -> (size, size, size) uint8, axes (z, y, x).

    The same :func:`billow_field` on a periodic cube, for the tilted
    integrator (§2.4): its z is height through a column (or the boil clock),
    and it tiles on all three axes. Not drawn in G1."""
    ax = np.arange(int(size), dtype=np.float64) / size
    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
    xyz = np.stack([x, y, z], axis=-1)
    return _to_u8(billow_field(xyz, seed=seed, base_cells=base_cells,
                               octaves=octaves, warp=warp))


__all__ = [
    "BASE_CELLS", "BILLOW_SEED", "perlin", "perlin_fbm", "worley_f1",
    "worley_domes", "billow_field", "bake_billow_rgba", "bake_billow_3d",
]
