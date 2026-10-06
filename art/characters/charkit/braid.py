"""Braid, lace and cord for dress uniforms (numpy + bpy, Blender 4.5).

Flat lace (frogging across a chest, chevrons above a cuff, knots on a thigh, piping on a seam,
edging) is carried as COLOUR: a `braid` vertex attribute in the kit's line encoding
(`1 - min(d, SEAM_CAP) / SEAM_CAP`, d = distance to the lace's centre line), which
`dressmat.mat_wool` turns into gold lace of a given width with a little relief. That survives
the game step's 6 mm remesh and 1024 px bake, where thin separate tubes would not. Free-hanging
braid (cords, tassels) is real geometry with real thickness.

Lace is drawn as polylines in a 2-D PROJECTION of the garment: (x, z) seen from the front or the
back, or (y, z) from the side, so a pattern is laid out the way the reference sheet shows it.

  polyline_dist(Q, lines)        distance from 2-D points to a set of polylines
  frog_rows(...)                 a hussar chest: rows of lace with loops at both ends
  hungarian_knot(...)            the trefoil knot on a thigh
  trefoil(...)                   a small three-loop knot (back seams, the seat)
  chevrons(...)                  inverted Vs stacked above a pointed cuff
  line_value(d)                  distance -> the attribute value
  cord(...) / tassel(...)        hanging braid: a twisted cord, a fringed tassel
"""
import math

import numpy as np

import kit
from kit import SEAM_CAP, TAU
from dresswear import smooth_path, swept


def line_value(d):
    return 1.0 - np.minimum(np.abs(d), SEAM_CAP) / SEAM_CAP


def polyline_dist(Q, lines, chunk=20000):
    """Distance from points Q (n, 2) to the nearest segment of any polyline in `lines`
    (a list of (k, 2) arrays)."""
    Q = np.asarray(Q, float).reshape(-1, 2)
    segs = [np.asarray(l, float) for l in lines if len(l) > 1]
    if not segs:
        return np.full(len(Q), 1.0)
    A = np.vstack([s[:-1] for s in segs])
    B = np.vstack([s[1:] for s in segs])
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12)
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        # cull segments far from this chunk's box
        bmin, bmax = q.min(axis=0) - SEAM_CAP, q.max(axis=0) + SEAM_CAP
        near = ~((np.maximum(A[:, 0], B[:, 0]) < bmin[0]) | (np.minimum(A[:, 0], B[:, 0]) > bmax[0])
                 | (np.maximum(A[:, 1], B[:, 1]) < bmin[1]) | (np.minimum(A[:, 1], B[:, 1]) > bmax[1]))
        if not near.any():
            out[i:i + chunk] = 1.0
            continue
        a, ab, l2 = A[near], AB[near], L2[near]
        d = np.full(len(q), 1.0)
        for j in range(0, len(a), 400):
            aq = q[:, None, :] - a[None, j:j + 400]
            k = np.clip(np.einsum("nmk,mk->nm", aq, ab[j:j + 400]) / l2[None, j:j + 400], 0.0, 1.0)
            dd = np.linalg.norm(aq - k[..., None] * ab[None, j:j + 400], axis=2).min(axis=1)
            d = np.minimum(d, dd)
        out[i:i + chunk] = d
    return out


def circle(c, r, n=24, a0=0.0, a1=TAU, ry=None):
    al = np.linspace(a0, a1, n)
    return np.column_stack([c[0] + r * np.cos(al), c[1] + (ry or r) * np.sin(al)])


def loop_end(x, z, sgn, size):
    """The loop at the end of a row of frogging: an oval eye pointing outwards (sgn +-1)."""
    c = (x + sgn * size, z)
    return circle(c, size, 20, 0.0, TAU, ry=0.55 * size)


def frog_rows(z0, z1, n, half_width, loop=0.010, double=0.0):
    """Rows of hussar frogging across a chest, as (x, z) polylines: `n` rows from z0 to z1, each
    from -w to +w (`half_width(z)` or a number), with an oval loop at each end. `double` > 0
    draws each row as two parallel cords that far apart."""
    lines = []
    for z in np.linspace(z0, z1, n):
        w = half_width(z) if callable(half_width) else half_width
        offs = (-0.5 * double, 0.5 * double) if double else (0.0,)
        for dz in offs:
            lines.append(np.array([[-w, z + dz], [w, z + dz]]))
        for sgn in (-1, 1):
            lines.append(loop_end(sgn * w, z, sgn, loop))
    return lines


def trefoil(c, size, up=1.0):
    """A small three-loop knot centred at c (x, z): two side loops and one on top (`up` = -1
    hangs the third loop down)."""
    x, z = c
    s = size
    return [circle((x - 0.9 * s, z), 0.55 * s, 18), circle((x + 0.9 * s, z), 0.55 * s, 18), circle((x, z + up * 0.95 * s), 0.6 * s, 18)]


def hungarian_knot(top, height, width):
    """The Austrian / Hungarian knot on the front of a thigh, as (x, z) polylines: two cords
    coming down from `top` (x, z), a pair of loops, crossing into one large loop and a pointed
    end, `height` long and `width` across."""
    x, z = top
    h, w = height, width
    pts_l = np.array([[x - 0.15 * w, z], [x - 0.18 * w, z - 0.12 * h], [x - 0.48 * w, z - 0.28 * h], [x - 0.36 * w, z - 0.44 * h],
                      [x - 0.05 * w, z - 0.40 * h], [x + 0.30 * w, z - 0.58 * h], [x + 0.36 * w, z - 0.76 * h], [x, z - h]])
    pts_r = pts_l.copy()
    pts_r[:, 0] = 2 * x - pts_r[:, 0]
    out = []
    for p in (pts_l, pts_r):
        q = np.column_stack([p[:, 0], np.zeros(len(p)), p[:, 1]])
        c = smooth_path(q, 60)
        out.append(c[:, [0, 2]])
    return out


def chevrons(s_point, t_point, half_s, drop, n, gap):
    """Inverted Vs in a limb's developed (s, t) coordinates: the first has its point at
    (s_point, t_point) and its arms `half_s` to either side, `drop` lower; the next `gap`
    above it, n of them."""
    out = []
    for i in range(n):
        tp = t_point + i * gap
        out.append(np.array([[s_point - half_s, tp - drop], [s_point, tp], [s_point + half_s, tp - drop]]))
    return out


# ------------------------------------------------------------------ hanging braid
def cord(name, pts, r=0.0035, n=None, twist=True, mat=None, coll="Garment"):
    """A twisted cord along control points `pts`: a capped tube whose surface carries the
    plait as relief (`twist`: a two-strand helix)."""
    pts = np.asarray(pts, float)
    L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    n = n or max(8, int(L / 0.003))
    c = smooth_path(pts, n)
    pitch = 5.0 * r
    disp = (lambda u, al: 0.18 * r * np.cos(2.0 * al - u * L * TAU / pitch)) if twist else None
    return swept(name, c, r, n_u=10, disp=disp, mat=mat, coll=coll, attrs=dict(braid=lambda u, al: np.ones(u.shape)))


def tassel(name, top, length, r_head, r_skirt, axis=(0.0, 0.0, -1.0), fringe=24, mat=None, coll="Garment"):
    """A tassel hanging from `top` along `axis`: a small round head, a waist, and a flared
    fringed skirt (`fringe` grooves round it). One closed object."""
    top, ax = np.asarray(top, float), kit.unit(np.asarray(axis, float))
    u = np.linspace(0.0, 1.0, 26)
    prof = np.interp(u, [0.0, 0.06, 0.18, 0.28, 0.34, 0.45, 0.80, 0.97, 1.0],
                     [0.35 * r_head, r_head, r_head, 0.65 * r_head, 0.8 * r_head, 0.75 * r_skirt, r_skirt, 0.9 * r_skirt, 0.3 * r_skirt])
    pts = top[None, :] + (u * length)[:, None] * ax[None, :]
    skirt = lambda uu, al: np.where(uu > 0.42, -0.22 * r_skirt * (0.5 + 0.5 * np.cos(fringe * al)) * np.clip((uu - 0.42) / 0.1, 0, 1), 0.0)
    return swept(name, pts, prof, n_u=max(24, fringe * 2), disp=skirt, mat=mat, coll=coll)
