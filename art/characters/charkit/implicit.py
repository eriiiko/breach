"""Implicit (signed-distance) shapes meshed to one closed surface (numpy; bpy only to emit).

For organic parts that lofts cannot make in one piece -- a hand whose fingers grow out of the
palm with webbing and no step -- the part is a signed-distance field: primitives (tapered
capsules of elliptical section, ellipsoids) combined with a SMOOTH union, so every junction is
a fillet. `mesh_field` samples the field on a grid, extracts the zero surface with surface nets
(one vertex per cell the surface crosses, one quad per grid edge it crosses: a closed quad mesh
for a field that is positive on the grid's border) and projects every vertex onto the true
surface by Newton steps along the gradient, so the mesh is as smooth as the field, not the grid.

Distances are in the field's own frame; the caller maps the vertices to the world.
"""
import numpy as np


def smin(a, b, k):
    """Polynomial smooth minimum (a smooth union of two distances): a fillet of width ~k."""
    if np.ndim(k) == 0 and k <= 0:
        return np.minimum(a, b)
    k = np.maximum(k, 1e-9)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def smax(a, b, k):
    """Smooth maximum (a smooth intersection / subtraction)."""
    return -smin(-a, -b, k)


def _perp_frame(T, hint):
    """Unit (U, V) across a unit axis T: V along `hint` made perpendicular, U = V x T."""
    V = hint - (hint @ T) * T
    n = np.linalg.norm(V)
    if n < 1e-9:
        V = np.array([1.0, 0.0, 0.0]) if abs(T[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        V = V - (V @ T) * T
        n = np.linalg.norm(V)
    V = V / n
    return np.cross(V, T), V


def capsule(P, c0, c1, r0, r1, hint, ratio=1.0, n=2.0, back=None):
    """A tapered capsule from c0 to c1 whose section is a superellipse: half-width r (across,
    along U) and half-depth r*ratio (along V, the `hint` side; `back` = the depth ratio on
    the -V side if different), r running linearly r0 -> r1; the ends rounded. Approximate
    distance (exact on the surface, scaled near it), negative inside."""
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    ax = c1 - c0
    ln = max(float(np.linalg.norm(ax)), 1e-9)
    T = ax / ln
    U, V = _perp_frame(T, np.asarray(hint, float))
    v = P - c0
    w = v @ T
    s = np.clip(w / ln, 0.0, 1.0)
    along = w - s * ln
    q = v - w[..., None] * T
    x, y = q @ U, q @ V
    r = r0 + (r1 - r0) * s
    rv = r * np.where(y >= 0.0, ratio, ratio if back is None else back)
    rc = np.minimum(r, rv)
    rho = (np.abs(x / r) ** n + np.abs(y / rv) ** n + np.abs(along / rc) ** n) ** (1.0 / n)
    return (rho - 1.0) * rc


def ellipsoid(P, c, axes, radii):
    """An ellipsoid: centre c, orthonormal axes (3, 3) (rows), radii (3,). Approximate distance."""
    q = (P - np.asarray(c, float)) @ np.asarray(axes, float).T
    rad = np.asarray(radii, float)
    rho = np.sqrt(np.sum((q / rad) ** 2, axis=-1))
    return (rho - 1.0) * rad.min()


def chain(P, nodes, hint, k=0.0015, ratio=1.0, n=2.0, back=None):
    """A chain of tapered capsules through nodes [(centre, radius), ...] (a finger, a palm),
    joined by a smooth union of width k (no crease at a bent joint). `hint` may be one vector
    or one per segment."""
    d = None
    hints = hint if np.ndim(hint) == 2 else [hint] * (len(nodes) - 1)
    for (c0, r0), (c1, r1), h in zip(nodes[:-1], nodes[1:], hints):
        e = capsule(P, c0, c1, r0, r1, h, ratio, n, back)
        d = e if d is None else smin(d, e, k)
    return d


# ---------------------------------------------------------------------------- meshing
def surface_nets(F, origin, h):
    """The zero surface of a sampled field F (nx, ny, nz; negative inside, positive on the
    border) as (verts (m, 3), quads (q, 4)), outward-facing, in the field's frame
    (grid point (i, j, k) at origin + h * (i, j, k))."""
    F = np.asarray(F, np.float64)
    nx, ny, nz = F.shape
    ins = F < 0.0
    acc = np.zeros((nx - 1, ny - 1, nz - 1, 3))
    cnt = np.zeros((nx - 1, ny - 1, nz - 1))
    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = np.stack([I, J, K], axis=-1).astype(np.float64)
    edges = []
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
        f0, f1 = F[tuple(sl0)], F[tuple(sl1)]
        e = ins[tuple(sl0)] != ins[tuple(sl1)]
        t = np.where(e, f0 / np.where(e, f0 - f1, 1.0), 0.0)
        p = base[tuple(sl0)].copy()
        p[..., ax] += t
        p *= e[..., None]
        others = [a for a in range(3) if a != ax]
        for d1 in (0, 1):
            for d2 in (0, 1):
                sl = [slice(None)] * 3
                sl[others[0]] = slice(d1, d1 + (F.shape[others[0]] - 1))
                sl[others[1]] = slice(d2, d2 + (F.shape[others[1]] - 1))
                acc += p[tuple(sl)]
                cnt += e[tuple(sl)]
        edges.append((ax, e, ins[tuple(sl0)]))
    active = cnt > 0
    vid = np.full(active.shape, -1, np.int64)
    vid[active] = np.arange(int(active.sum()))
    verts = acc[active] / cnt[active][:, None]
    quads = []
    for ax, e, start_in in edges:
        o1, o2 = [a for a in range(3) if a != ax]
        # (o1, o2) is a right-handed pair with ax: (1,2) for x, (2,0) for y, (0,1) for z
        if (o1, o2) == (0, 2):
            o1, o2 = 2, 0
        sel = e.copy()
        # an edge needs all four neighbouring cells: drop those on the border along o1 / o2
        s = [slice(None)] * 3
        s[o1] = slice(0, 1)
        sel[tuple(s)] = False
        s[o1] = slice(e.shape[o1] - 1, None)
        sel[tuple(s)] = False
        s = [slice(None)] * 3
        s[o2] = slice(0, 1)
        sel[tuple(s)] = False
        s[o2] = slice(e.shape[o2] - 1, None)
        sel[tuple(s)] = False
        idx = np.argwhere(sel)
        if not len(idx):
            continue
        si = start_in[sel]

        def cell(d1, d2):
            c = idx.copy()
            c[:, o1] -= d1
            c[:, o2] -= d2
            return vid[c[:, 0], c[:, 1], c[:, 2]]

        q = np.stack([cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)], axis=1)
        q = np.where(si[:, None], q, q[:, ::-1])
        quads.append(q)
    quads = np.vstack(quads)
    return np.asarray(origin, float) + h * verts, quads


def project(field, V, h, iters=4):
    """Move points onto the field's zero surface (Newton steps along the numerical gradient)."""
    V = np.array(V, float)
    e = 0.25 * h
    for _ in range(iters):
        f = field(V)
        g = np.stack([(field(V + e * u) - field(V - e * u)) / (2 * e) for u in np.eye(3)], axis=1)
        gg = np.maximum(np.sum(g * g, axis=1), 1e-12)
        step = (f / gg)[:, None] * g
        n = np.linalg.norm(step, axis=1, keepdims=True)
        V -= step * np.minimum(1.0, h / np.maximum(n, 1e-12))  # never jump farther than a cell
    return V


def mesh_field(field, lo, hi, h, chunk=200000, border=True):
    """Mesh the zero surface of `field(P (n, 3)) -> d (n,)` inside the box lo..hi at pitch h.
    Returns (verts, quads) facing out: closed when `border` (the field is forced positive on the
    box's faces, so a part inside the box is sealed), an open patch ending at the box when not."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    gx, gy, gz = (lo[i] + h * np.arange(n[i]) for i in range(3))
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    F = np.concatenate([field(P[i:i + chunk]) for i in range(0, len(P), chunk)]).reshape(n)
    if border:
        F[0], F[-1], F[:, 0], F[:, -1], F[:, :, 0], F[:, :, -1] = (np.abs(F[0]) + h, np.abs(F[-1]) + h, np.abs(F[:, 0]) + h,
                                                                    np.abs(F[:, -1]) + h, np.abs(F[:, :, 0]) + h, np.abs(F[:, :, -1]) + h)
    V, Q = surface_nets(F, lo, h)
    return project(field, V, h), Q


def boundary_mask(Q, nv):
    """Vertices on an open edge of a quad mesh."""
    E = np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    E = np.sort(E, axis=1)
    key = E[:, 0].astype(np.int64) * nv + E[:, 1]
    u, c = np.unique(key, return_counts=True)
    b = u[c == 1]
    m = np.zeros(nv, bool)
    m[b // nv] = True
    m[b % nv] = True
    return m


def taubin(V, Q, iters=10, lam=0.5, mu=-0.53, fixed=None):
    """Taubin smoothing (a low-pass filter that keeps the volume): removes the fine ripple a
    sampled field leaves, keeps the form. `fixed` vertices do not move."""
    V = np.array(V, float)
    E = np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    deg = np.bincount(E.ravel(), minlength=len(V)).astype(float)
    free = np.ones(len(V), bool) if fixed is None else ~fixed
    for _ in range(iters):
        for f in (lam, mu):
            S = np.zeros_like(V)
            np.add.at(S, E[:, 0], V[E[:, 1]])
            np.add.at(S, E[:, 1], V[E[:, 0]])
            L = S / np.maximum(deg, 1.0)[:, None] - V
            V[free] += f * L[free]
    return V
