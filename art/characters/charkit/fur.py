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


def fur_roll(name, pts, r, seed=3, flat=0.85, density=14000.0, tuft=(0.004, 0.008), pile=(0.06, 0.20), up=None, n=None,
             taper=0.0, scale=None, soft=None, mat=None, coll="Garment"):
    """A roll of fur `r` thick along control points `pts` (a smooth curve through them),
    squashed to `flat` across, covered in random tufts (`density` per m^2, sizes `tuft` in m,
    heights `pile` as fractions of r). `taper` thins both ends over that fraction of the
    length. `scale` (one per control point) varies the radius along the roll, and `flat` may
    be one per control point too. `soft` (a dict of `pile_field` arguments, at least `amp`),
    if given, replaces the random tufts by a fine directional pile. Capped: one closed object."""
    c = smooth_path(pts, n or max(12, int(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1).sum() / 0.004)))
    L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    circ = TAU * r
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, len(c))
    p = np.asarray(pts, float)
    uc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    along = lambda a: np.interp(u, uc / uc[-1], np.asarray(a, float))  # a per-control-point value along the resampled curve
    radii = np.full(len(c), r)
    if taper > 0:
        radii = r * np.clip(np.minimum(u, 1 - u) / taper, 0.25, 1.0) ** 0.5
    if scale is not None:
        radii = radii * along(scale)
    if np.ndim(flat) == 1 and len(flat) == len(p):
        flat = along(flat)
    n_u = max(16, int(circ * float(np.max(radii)) / r / 0.004))
    if up is not None and np.ndim(up) == 2:  # one per control point: carried along the resampled curve
        up = np.column_stack([along(np.asarray(up, float)[:, k]) for k in range(3)])
    if soft is not None:
        pf = pile_field(seed, L, circ, **soft)
        return swept(name, c, radii, n_u=n_u, flat=flat, up=up, disp=lambda UU, AL: pf(UU, AL, r), mat=mat, coll=coll)
    tf = _tufts(rng, L, circ, density, tuft, (pile[0] * r, pile[1] * r))
    field = _tuft_field(tf, L, circ)
    return swept(name, c, radii, n_u=n_u, flat=flat, up=up, disp=lambda UU, AL: field(UU, AL, r) - 0.2 * r, mat=mat, coll=coll)


def pile_field(seed, L, circ, amp, across=(0.006, 0.014), along=(0.012, 0.040), lean=0.35, waves=48, spike=1.6):
    """A fine, directional pile as a displacement (m) over a tube's developed surface (s along,
    a around): a sum of `waves` random plane waves, short ACROSS the pile (`across`
    wavelengths, m) and long ALONG it (`along`), the pile direction leaning `lean` (radians)
    off the tube's length, shaped to soft narrow ridges (`spike` > 1). Vectorised: cheap
    on fine grids. Returns f(UU, AL, r) for `fur_roll(..., pile_fn=)` and `plume`."""
    rng = np.random.default_rng(seed)
    la = rng.uniform(*across, waves)
    ll = rng.uniform(*along, waves)
    th = lean + rng.normal(0.0, 0.18, waves)
    ph = rng.uniform(0.0, TAU, waves)
    # wave numbers in (s, a), with a wrapped round the tube: an integer count round it
    ka = np.round(circ / la) * TAU / circ
    ks = TAU / ll
    ca, sa = np.cos(th), np.sin(th)
    k_s, k_a = ks * ca - ka * sa, ks * sa + ka * ca
    k_a = np.round(k_a * circ / TAU) * TAU / circ  # seamless round the tube
    w = rng.uniform(0.5, 1.0, waves)
    w = w / w.sum()
    norm = 2.2 * math.sqrt(float((w ** 2).sum()) / 2.0)  # ~ the sum's spread: g spans 0..1

    def f(UU, AL, r):
        s, a = UU * L, AL / TAU * circ
        acc = np.zeros(UU.shape)
        for i in range(waves):
            acc += w[i] * np.sin(k_s[i] * s + k_a[i] * a + ph[i])
        g = np.clip(0.5 + 0.5 * acc / norm, 0.0, 1.0) ** spike
        return amp * (g - 0.5)

    return f


def plume_soft(name, spec, seed=5, mat=None, coll="Plumes", grid=0.0016):
    """A SOFT feather plume: a smooth tapered ovoid on a curved stem (narrow at the socket,
    widest about two thirds up, a rounded tip), its edge broken by a fine hairy pile (barbs a
    few millimetres across lying up the plume, `pile_field`), no tufts. `spec` as `plume`,
    plus optional `pile` (m, the barbs' height). From above it reads as a round fuzzy tuft."""
    c = smooth_path(spec["path"], max(80, int(np.linalg.norm(np.diff(np.asarray(spec["path"], float), axis=0), axis=1).sum() / grid)))
    L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    r = spec["r"]
    u = np.linspace(0.0, 1.0, len(c))
    pu, pr = zip(*spec.get("radius", ((0.0, 0.10), (0.06, 0.26), (0.20, 0.55), (0.40, 0.84), (0.60, 0.98), (0.72, 1.0), (0.84, 0.90),
                                      (0.93, 0.66), (0.98, 0.36), (1.0, 0.06))))
    radii = r * np.interp(u, pu, pr)
    circ = TAU * r
    pile = spec.get("pile", 0.0045)
    field = pile_field(seed, L, circ, pile, across=(0.0030, 0.0065), along=(0.014, 0.045), lean=0.0, waves=56, spike=2.2)
    swell = lambda UU: np.clip(np.interp(UU, pu, pr) / 0.5, 0.0, 1.0)
    n_u = max(48, int(circ / grid))
    return swept(name, c, radii, n_u=n_u, disp=lambda UU, AL: field(UU, AL, r) * swell(UU),
                 attrs=dict(tip=lambda UU, AL: UU), mat=mat, coll=coll)


def plume(name, spec, seed=5, mat=None, coll="Plumes"):
    """A feather plume: a stem curving through `spec["path"]` (base -> tip, world points),
    its radius `spec["radius"]` along the length as ((u, fraction of `r`), ...) with `r` the
    widest, covered in feather tufts lying along it. `tip` attribute = u, so the material
    can colour the last part of it (a red-tipped plume)."""
    c = smooth_path(spec["path"], 140)
    L = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    r = spec["r"]
    u = np.linspace(0.0, 1.0, len(c))
    pu, pr = zip(*spec.get("radius", ((0.0, 0.16), (0.10, 0.30), (0.25, 0.55), (0.45, 0.88), (0.65, 1.0), (0.82, 0.88), (0.93, 0.60),
                                      (0.985, 0.25), (1.0, 0.05))))
    radii = r * np.interp(u, pu, pr)
    circ = TAU * r
    rng = np.random.default_rng(seed)
    tf = _tufts(rng, L, circ, spec.get("density", 30000.0), (0.0025, 0.0050), (0.07 * r, 0.22 * r))
    field = _tuft_field(tf, L, circ, k_up=1.0)
    swell = lambda UU: np.interp(UU, pu, pr)
    n_u = max(32, int(circ / 0.0028))
    return swept(name, c, radii, n_u=n_u, disp=lambda UU, AL: (field(UU, AL, r) - 0.15 * r) * swell(UU),
                 attrs=dict(tip=lambda UU, AL: UU), mat=mat, coll=coll)


def fur_patch(name, loft, phi_of, t_of, height, seed=4, lift=0.002, edge=0.25, density=16000.0, tuft=(0.004, 0.009), pile=(0.10, 0.30),
              thick=0.004, mat=None, coll="Garment"):
    """A band of fur lying on a loft (a shawl collar, a wide facing): the `dresswear.loft_region`
    patch (phi_of, t_of over the unit square) raised `height` off the surface, rounded down to
    the surface over the last `edge` of each side, covered in random tufts (`density` per m^2,
    `tuft` sizes, `pile` heights as fractions of `height`), `thick` inwards. Closed."""
    import kit
    from dresswear import loft_region
    rng = np.random.default_rng(seed)

    def disp(g):
        k = np.minimum(np.clip(np.minimum(g.v, 1.0 - g.v) / edge, 0.0, 1.0), np.clip(np.minimum(g.u, 1.0 - g.u) / edge, 0.0, 1.0))
        dome = np.sqrt(k * (2.0 - k))
        P = g.P.reshape(-1, 3)
        lo, hi = P.min(axis=0) - 0.01, P.max(axis=0) + 0.01
        area = float(np.prod(np.sort(hi - lo)[1:]))
        n = max(16, int(area * density))
        C = rng.uniform(lo, hi, (n, 3))
        w = rng.uniform(*tuft, n)
        h = rng.uniform(pile[0] * height, pile[1] * height, n)
        out = np.zeros(len(P))
        for i in range(0, n, 256):
            d2 = ((P[:, None, :] - C[None, i:i + 256]) ** 2).sum(axis=2)
            out += (h[None, i:i + 256] * np.exp(-d2 / w[None, i:i + 256] ** 2)).sum(axis=1)
        return (height + out.reshape(g.P.shape[:-1])) * dome

    obj = loft_region(name, loft, phi_of, t_of, offset=lift, disp=disp, mat=mat, coll=coll, res=min(kit.RES, 0.004))
    return kit.solid(obj, thick, bevel=0.0)
