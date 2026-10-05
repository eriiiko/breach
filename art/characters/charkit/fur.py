"""Fur trim and feather plumes (numpy + bpy, Blender 4.5): soft, tufted closed shapes.

Both are tubes (`dresswear.swept`) whose surface is pushed out by many small random tufts,
so the outline reads soft and broken at any distance, with real thickness for the game
step's 6 mm remesh; the strands themselves are left to the material (`dressmat.mat_fur`,
`mat_feather`).

  fur_roll(name, pts, r, ...)     a roll of fur along a path: a collar, a hem, a front edge
  plume(name, spec, ...)          a feather plume on a curved stem, its `tip` attribute 0 at
                                  the base -> 1 at the tip, for a tipped plume's colour
"""
import math

import numpy as np

from dresswear import smooth_path, swept
from kit import TAU


def _tufts(rng, L, circ, density, size, height):
    n = max(8, int(L * circ * density))
    return dict(s=rng.uniform(0.0, L, n), a=rng.uniform(0.0, circ, n), w=rng.uniform(*size, n), h=rng.uniform(*height, n))


def _tuft_field(tf, L, circ, k_up=0.0):
    """Sum of elliptical bumps on the developed (s along, a around) surface; `k_up` > 0
    stretches each bump along the length (feathers lie along the plume)."""

    def f(UU, AL, r_local):
        s, a = UU * L, AL / TAU * circ
        out = np.zeros(UU.shape)
        for i in range(len(tf["s"])):
            ds = s - tf["s"][i]
            da = (a - tf["a"][i] + 0.5 * circ) % circ - 0.5 * circ
            w = tf["w"][i]
            out += tf["h"][i] * np.exp(-(ds / (w * (1.0 + k_up))) ** 2 - (da / w) ** 2)
        return out

    return f


def fur_roll(name, pts, r, seed=3, flat=0.85, density=9000.0, tuft=(0.004, 0.009), pile=(0.15, 0.45), up=None, n=None,
             taper=0.0, mat=None, coll="Garment"):
    """A roll of fur `r` thick along control points `pts` (a smooth curve through them),
    squashed to `flat` across, covered in random tufts (`density` per m^2, sizes `tuft` in m,
    heights `pile` as fractions of r). `taper` thins both ends over that fraction of the
    length. Capped: one closed object."""
    c = smooth_path(pts, n or max(12, int(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1).sum() / 0.004)))
    L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    circ = TAU * r
    rng = np.random.default_rng(seed)
    tf = _tufts(rng, L, circ, density, tuft, (pile[0] * r, pile[1] * r))
    field = _tuft_field(tf, L, circ)
    u = np.linspace(0.0, 1.0, len(c))
    radii = np.full(len(c), r)
    if taper > 0:
        radii = r * np.clip(np.minimum(u, 1 - u) / taper, 0.25, 1.0) ** 0.5
    n_u = max(16, int(circ / 0.004))
    return swept(name, c, radii, n_u=n_u, flat=flat, up=up, disp=lambda UU, AL: field(UU, AL, r) - 0.15 * r, mat=mat, coll=coll)


def plume(name, spec, seed=5, mat=None, coll="Plumes"):
    """A feather plume: a stem curving through `spec["path"]` (base -> tip, world points),
    its radius `spec["radius"]` along the length as ((u, fraction of `r`), ...) with `r` the
    widest, covered in feather tufts lying along it. `tip` attribute = u, so the material
    can colour the last part of it (a red-tipped plume)."""
    c = smooth_path(spec["path"], 90)
    L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    r = spec["r"]
    u = np.linspace(0.0, 1.0, len(c))
    pu, pr = zip(*spec.get("radius", ((0.0, 0.10), (0.08, 0.25), (0.35, 0.80), (0.65, 1.0), (0.88, 0.80), (0.97, 0.45), (1.0, 0.08))))
    radii = r * np.interp(u, pu, pr)
    circ = TAU * r
    rng = np.random.default_rng(seed)
    tf = _tufts(rng, L, circ, spec.get("density", 2600.0) / max(r, 1e-3) * 0.05, (0.006, 0.014), (0.10 * r, 0.32 * r))
    field = _tuft_field(tf, L, circ, k_up=1.6)
    swell = lambda UU: np.interp(UU, pu, pr)
    n_u = max(24, int(circ / 0.0045))
    return swept(name, c, radii, n_u=n_u, disp=lambda UU, AL: (field(UU, AL, r) - 0.12 * r) * swell(UU),
                 attrs=dict(tip=lambda UU, AL: UU), mat=mat, coll=coll)
