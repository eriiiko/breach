"""Parametric modelling kit for scripted characters (Blender 4.5, numpy + bpy).

Three ideas carry the whole model:

* `Loft`   -- a limb/torso/helmet as a smooth tube: superellipse cross-sections
              interpolated along a centre line. It is an analytic surface
              `(phi, t) -> point, normal`, so everything else can be placed ON it.
* anchors  -- `OnLoft`, `OnEllipsoid`, `OnPlane`: a local metric (s, t) frame on a
              surface. `plate()` and `box_at()` build armour, straps, pockets and
              pouches in that frame, so they hug the body they sit on.
* fields   -- per-vertex callbacks on the loft grid: `disp` (authored folds),
              `attrs` (float masks the materials read: `flex`, `seam`).

All geometry is emitted in WORLD coordinates with objects at identity, so a
Mirror modifier about the origin gives the other half of the character.
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix


def _yield_the_machine():
    """A headless build drops to the lowest processor priority (Windows), so a game being played on
    the same machine keeps its frame rate. Incident (#33, 2026-10-04): all-core character builds
    pushed the game's sim tick past its period, and the game's catch-up loop turned that into
    200 ms frames; at idle priority the same load left every frame under 62 ms."""
    if os.name != "nt" or not bpy.app.background:
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000040)  # IDLE_PRIORITY_CLASS
    except Exception:  # a build must never fail over its priority
        pass


_yield_the_machine()

RES = 0.004  # grid pitch in metres; build.py overrides it for drafts
TAU = 2.0 * math.pi
SEAM_CAP = 0.02  # the `seam` attribute stores 1 - min(d, SEAM_CAP) / SEAM_CAP


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def wrap(a):
    """Angle difference wrapped to [-pi, pi)."""
    return (np.asarray(a, float) + math.pi) % TAU - math.pi


def spline(tk, yk, tq, monotone=False):
    """Cubic Hermite through knots (tk, yk) evaluated at tq.

    Catmull-Rom tangents by default; `monotone` uses harmonic-mean tangents, so a
    radius profile never overshoots its knots.
    """
    tk = np.asarray(tk, float)
    yk = np.asarray(yk, float)
    tq = np.atleast_1d(np.asarray(tq, float))
    flat = yk.ndim == 1
    y = yk[:, None] if flat else yk
    h = np.diff(tk)[:, None]
    d = np.diff(y, axis=0) / h
    m = np.empty_like(y)
    m[0], m[-1] = d[0], d[-1]
    if len(tk) > 2:
        if monotone:
            same = d[:-1] * d[1:] > 0
            m[1:-1] = np.where(same, 2.0 * d[:-1] * d[1:] / np.where(same, d[:-1] + d[1:], 1.0), 0.0)
        else:
            m[1:-1] = (d[:-1] * h[1:] + d[1:] * h[:-1]) / (h[:-1] + h[1:])
    i = np.clip(np.searchsorted(tk, tq, side="right") - 1, 0, len(tk) - 2)
    hh = (tk[i + 1] - tk[i])[:, None]
    s = np.clip((tq - tk[i])[:, None] / hh, 0.0, 1.0)
    out = ((2 * s**3 - 3 * s**2 + 1) * y[i] + (s**3 - 2 * s**2 + s) * hh * m[i]
           + (-2 * s**3 + 3 * s**2) * y[i + 1] + (s**3 - s**2) * hh * m[i + 1])
    return out[:, 0] if flat else out


# --------------------------------------------------------------------------- loft
class Loft:
    """Tube surface: superellipse sections along a centre line.

    A ring is a dict: `p` centre, `a` half-width (along e1), `b` or `bf`/`bb`
    half-depth towards / away from `front` (along e2), `n` superellipse exponent
    (or a `(front, back)` pair), optional `t` to pin the section plane's normal.
    `phi` is the polar angle in the section: 0 at +e1, 90 deg at the front.
    """

    def __init__(self, rings=None, front=(0.0, -1.0, 0.0), fn=None, length=None):
        self.front = unit(np.asarray(front, float))
        self.fn = fn
        if fn is not None:
            self.L = float(length)
            return
        P = np.array([r["p"] for r in rings], float)
        self.tk = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        self.L = float(self.tk[-1])
        self.Pk = P
        par = []
        for r in rings:
            b = r.get("b")
            bf = r.get("bf", b)
            bb = r.get("bb", b if b is not None else bf)
            n = r.get("n", 2.0)
            nf, nb = n if isinstance(n, tuple) else (n, n)
            par.append((r["a"], bf, bb, nf, nb))
        self.park = np.array(par, float)
        eps = 1e-4 * self.L
        T = unit(spline(self.tk, P, np.clip(self.tk + eps, 0, self.L)) - spline(self.tk, P, np.clip(self.tk - eps, 0, self.L)))
        for i, r in enumerate(rings):
            if "t" in r:
                T[i] = unit(np.asarray(r["t"], float))
        self.Tk = T

    def t_ring(self, i):
        return float(self.tk[i])

    def t_at_z(self, z):
        """Arclength where the centre line reaches height z (centre z must rise with t)."""
        ts = np.linspace(0.0, self.L, 600)
        zs = self.frames(ts)[0][:, 2]
        return float(np.interp(z, zs, ts))

    def frames(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        if self.fn is not None:
            c, T, par = self.fn(t)
        else:
            c = spline(self.tk, self.Pk, t)
            T = unit(spline(self.tk, self.Tk, t))
            par = spline(self.tk, self.park, t, monotone=True)
        e2 = unit(self.front[None, :] - (T @ self.front)[:, None] * T)
        e1 = np.cross(T, e2)
        return c, T, e1, e2, par

    @staticmethod
    def _polar(phi, par):
        a, bf, bb, nf, nb = (np.maximum(par[:, k], 1e-6) for k in range(5))
        cs, sn = np.cos(phi), np.sin(phi)
        fr = sn >= 0
        b = np.where(fr, bf, bb)
        n = np.where(fr, nf, nb)
        return (np.abs(cs / a) ** n + np.abs(sn / b) ** n) ** (-1.0 / n), cs, sn

    def radius(self, phi, t):
        phi = np.atleast_1d(np.asarray(phi, float))
        return self._polar(phi, self.frames(t)[4])[0]

    def pos(self, phi, t):
        c, _, e1, e2, par = self.frames(t)
        r, cs, sn = self._polar(phi, par)
        return c + (r * cs)[:, None] * e1 + (r * sn)[:, None] * e2

    def pn(self, phi, t, offset=0.0):
        """Points and outward unit normals at flat arrays (phi, t); offset along the normal."""
        phi = np.atleast_1d(np.asarray(phi, float)).ravel()
        t = np.clip(np.atleast_1d(np.asarray(t, float)).ravel(), 0.0, self.L)
        if t.size == 1 and phi.size > 1:
            t = np.full(phi.shape, t[0])
        P = self.pos(phi, t)
        hp, ht = 2e-3, min(1e-3, 0.01 * self.L)
        dphi = self.pos(phi + hp, t) - self.pos(phi - hp, t)
        dt = self.pos(phi, np.clip(t + ht, 0, self.L)) - self.pos(phi, np.clip(t - ht, 0, self.L))
        N = unit(np.cross(dt, dphi))
        off = np.asarray(offset, float)
        return P + (off.ravel()[:, None] if off.ndim else off) * N, N


def loft_sdf(loft, P, m=600, chunk=3000):
    """Approximate signed distance (m, negative inside) from points `P` (n, 3) to a loft's
    surface, measured radially in the section plane that contains each point. Points
    beyond either end of the loft count as outside (+1). Good near a tube-like loft whose
    centre line does not fold back on itself (a sleeve, a leg)."""
    P = np.asarray(P, float).reshape(-1, 3)
    ts = np.linspace(0.0, loft.L, m)
    C, T, e1, e2, par = loft.frames(ts)
    reach = 1.6 * float(par[:, :3].max())
    out = np.ones(len(P))
    for i in range(0, len(P), chunk):
        D = P[i:i + chunk, None, :] - C[None]
        f = np.einsum("nmk,mk->nm", D, T)
        dist = np.linalg.norm(D, axis=2)
        k = np.argmin(np.abs(f) + 2.0 * np.maximum(dist - reach, 0.0), axis=1)
        n = np.arange(len(k))
        fk = f[n, k]
        q = D[n, k] - fk[:, None] * T[k]
        x1, x2 = np.einsum("nk,nk->n", q, e1[k]), np.einsum("nk,nk->n", q, e2[k])
        r = Loft._polar(np.arctan2(x2, x1), par[k])[0]
        valid = np.abs(fk) < 1.5 * loft.L / m + 1e-4
        out[i:i + chunk] = np.where(valid, np.hypot(x1, x2) - r, 1.0)
    return out


def smax(a, b, k):
    """Smooth maximum (polynomial): max(a, b) with a fillet of width k where they meet."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return a * (1.0 - h) + b * h + k * h * (1.0 - h) * 0.5


class Oval:
    """Superellipse outline drawn on a loft, in developed metric coordinates.

    `g < 0` inside. `taper` widens the top (t above centre) and narrows the bottom.
    """

    def __init__(self, loft, phi0, t0, hs, ht, n=3.0, taper=0.0, scale=1.0):
        self.loft, self.phi0, self.t0 = loft, phi0, t0
        self.hs, self.ht, self.n, self.taper = hs * scale, ht * scale, n, taper
        self.r = float(loft.radius(phi0, t0)[0])

    def scaled(self, k):
        o = Oval(self.loft, self.phi0, self.t0, self.hs, self.ht, self.n, self.taper, k)
        return o

    def _g(self, ds, dt):
        ds = ds / (1.0 + self.taper * dt / self.ht)
        return (np.abs(ds / self.hs) ** self.n + np.abs(dt / self.ht) ** self.n) ** (1.0 / self.n) - 1.0

    def g(self, phi, t):
        return self._g(wrap(phi - self.phi0) * self.r, t - self.t0)

    def snap(self, phi, t, mask):
        """Move the masked vertices radially onto the outline."""
        ds, dt = wrap(phi - self.phi0) * self.r, t - self.t0
        for _ in range(4):
            k = np.where(mask, 1.0 / np.maximum(self._g(ds, dt) + 1.0, 1e-6), 1.0)
            ds, dt = ds * k, dt * k
        return self.phi0 + ds / self.r, self.t0 + dt

    def outline(self, k=240, scale=1.0):
        al = np.linspace(0.0, TAU, k, endpoint=False)
        rho = (np.abs(np.cos(al) / self.hs) ** self.n + np.abs(np.sin(al) / self.ht) ** self.n) ** (-1.0 / self.n) * scale
        dt = rho * np.sin(al)
        ds = rho * np.cos(al) * (1.0 + self.taper * dt / self.ht)
        return self.phi0 + ds / self.r, self.t0 + dt


# ------------------------------------------------------------------------- meshes
def link(obj, coll="Marine"):
    c = bpy.data.collections.get(coll)
    if c is None:
        c = bpy.data.collections.new(coll)
        bpy.context.scene.collection.children.link(c)
    c.objects.link(obj)
    return obj


def new_mesh(name, verts, quads, tris=None, mat=None, coll="Marine", attrs=None, smooth=True):
    verts = np.asarray(verts, np.float32)
    quads = np.asarray(quads, np.int32).reshape(-1, 4)
    tris = np.zeros((0, 3), np.int32) if tris is None else np.asarray(tris, np.int32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    nq, nt = len(quads), len(tris)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(nq * 4 + nt * 3)
    me.loops.foreach_set("vertex_index", np.concatenate([quads.ravel(), tris.ravel()]))
    me.polygons.add(nq + nt)
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3]).astype(np.int32))
    me.update(calc_edges=True)
    me.validate(verbose=False)
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    for k, v in (attrs or {}).items():
        a = me.attributes.new(k, "FLOAT", "POINT")
        a.data.foreach_set("value", np.asarray(v, np.float32).ravel())
    if mat is not None:
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll)


class Grid:
    """What a `disp` / `attrs` / `offset` callback sees: (rows, cols) arrays."""

    def snap_phi(self, phi):
        return self.cols[np.argmin(np.abs(wrap(self.cols - phi)))]

    def snap_t(self, t):
        return self.rows[np.argmin(np.abs(self.rows - t))]


def loft_mesh(name, loft, t0=0.0, t1=None, phi0=0.0, phi1=TAU, res=None, rows=None, offset=0.0,
              disp=None, attrs=None, hole=None, keep=None, cap0=False, cap1=False, post=None, drop=None, mat=None, coll="Marine"):
    """Grid mesh over a loft. `hole`/`keep` cut an `Oval` (boundary verts snapped onto it)."""
    res = res or RES
    t1 = loft.L if t1 is None else t1
    closed = abs(phi1 - phi0 - TAU) < 1e-9
    if rows is None:
        rows = np.linspace(t0, t1, max(2, int(round((t1 - t0) / res))) + 1)
    rows = np.asarray(rows, float)
    probe_t = np.repeat(np.linspace(rows[0], rows[-1], 9), 12)
    probe_p = np.tile(np.linspace(phi0, phi1, 12), 9)
    nc = int(round((phi1 - phi0) * float(loft.radius(probe_p, probe_t).max()) / res))
    nc = max(8, nc + (-nc) % 4)
    cols = np.linspace(phi0, phi1, nc, endpoint=False) if closed else np.linspace(phi0, phi1, nc + 1)
    R, C = len(rows), len(cols)
    PH, TT = np.meshgrid(cols, rows)
    g = Grid()
    g.loft, g.rows, g.cols, g.res = loft, rows, cols, res
    shape = hole if hole is not None else keep
    good = None
    if shape is not None:
        val = shape.g(PH, TT)
        good = (val > 0) if hole is not None else (val < 0)
        PH, TT = shape.snap(PH, TT, ~good)
        g.rho = shape.g(PH, TT) + 1.0
    g.phi, g.t = PH, TT
    g.r = loft.radius(PH.ravel(), TT.ravel()).reshape(R, C)
    off = offset(g) if callable(offset) else offset
    P, N = loft.pn(PH.ravel(), TT.ravel(), np.asarray(off, float).ravel() if np.ndim(off) else off)
    g.P, g.N = P.reshape(R, C, 3), N.reshape(R, C, 3)
    if disp is not None:
        P = P + np.asarray(disp(g), float).ravel()[:, None] * N
    vattrs = {k: np.asarray(f(g), float).ravel() for k, f in (attrs or {}).items()}
    dropv = drop(P) if drop is not None else None
    if post is not None:  # a post marked `grid` sees the vertices as the (rows, cols, 3) grid
        P = post(P.reshape(R, C, 3)).reshape(-1, 3) if getattr(post, "grid", False) else post(P)
    idx = np.arange(R * C).reshape(R, C)
    if closed:
        nxt = np.roll(idx, -1, axis=1)
        faces = np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], axis=-1).reshape(-1, 4)
    else:
        faces = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
    if good is not None:
        faces = faces[good.ravel()[faces].any(axis=1)]
    if dropv is not None:
        faces = faces[~dropv[faces].all(axis=1)]
    tris = None
    for row, on, flip in ((idx[0], cap0, True), (idx[-1], cap1, False)):
        if not on:
            continue
        P = np.vstack([P, P[row].mean(axis=0)])
        fan = np.stack([row, np.full(C, len(P) - 1), np.roll(row, -1)], axis=-1)
        fan = fan[:, ::-1] if flip else fan
        tris = fan if tris is None else np.vstack([tris, fan])
        vattrs = {k: np.append(v, v[row].mean()) for k, v in vattrs.items()}
    used = np.zeros(len(P), bool)
    used[faces.ravel()] = True
    if tris is not None:
        used[tris.ravel()] = True
    remap = np.cumsum(used) - 1
    return new_mesh(name, P[used], remap[faces], None if tris is None else remap[tris], mat, coll,
                    {k: v[used] for k, v in vattrs.items()})


def seam_attr(g, phis=(), ts=(), extra=None):
    """`seam` attribute: lines at phi (optionally `(phi, t_lo, t_hi)`) and at t
    (optionally `(t, phi_centre, half_angle)`), snapped to grid lines so the
    interpolated distance reaches zero and the line has a constant width."""
    d = np.full(g.phi.shape, 1.0)
    for p in phis:
        p, lo, hi = p if isinstance(p, tuple) else (p, -1e9, 1e9)
        dd = np.abs(wrap(g.phi - g.snap_phi(p))) * g.r
        d = np.minimum(d, np.where((g.t >= lo) & (g.t <= hi), dd, 1.0))
    for t in ts:
        t, pc, hw = t if isinstance(t, tuple) else (t, 0.0, 9.0)
        dd = np.abs(g.t - g.snap_t(t))
        d = np.minimum(d, np.where(np.abs(wrap(g.phi - pc)) <= hw, dd, 1.0))
    if extra is not None:
        d = np.minimum(d, extra)
    return 1.0 - np.minimum(d, SEAM_CAP) / SEAM_CAP


def band_mask(g, sd):
    """Signed distance (m, negative inside) -> 0..1 mask with a one-cell ramp."""
    return np.clip(0.5 - sd / (2.0 * g.res), 0.0, 1.0)


def fold_set(rng, n, t, phi, ang=(0.0, 0.2), half=(0.05, 0.10), w=(0.006, 0.010), h=(0.003, 0.006)):
    return [dict(t=rng.uniform(*t), phi=rng.uniform(*phi), ang=rng.normal(*ang), half=rng.uniform(*half),
                 w=rng.uniform(*w), h=rng.uniform(*h)) for _ in range(n)]


def crumple_set(rng, n, t, phi=(0.0, TAU)):
    """The all-over small creasing of padded cloth: many short, shallow, mostly
    around-the-limb folds. Laid under the authored folds of `fold_set`."""
    return fold_set(rng, n, t, phi, (0.0, 0.55), (0.018, 0.050), (0.0035, 0.0070), (0.0012, 0.0030))


def fold_field(folds):
    """Authored cloth folds: each a tapered ridge with a trough beside it, laid on
    the developed (s, t) surface of the loft at angle `ang` to the around-direction."""

    def f(g):
        d = np.zeros(g.phi.shape)
        for fo in folds:
            ds = wrap(g.phi - fo["phi"]) * g.r
            dt = g.t - fo["t"]
            ca, sa = math.cos(fo["ang"]), math.sin(fo["ang"])
            along, perp = ds * ca + dt * sa, -ds * sa + dt * ca
            taper = np.cos(0.5 * math.pi * np.clip(np.abs(along) / fo["half"], 0.0, 1.0)) ** 2
            w = fo["w"]
            d += fo["h"] * taper * (np.exp(-(perp / w) ** 2) - 0.45 * np.exp(-((perp + 1.9 * w) / (1.3 * w)) ** 2))
        return d

    return f


# ------------------------------------------------------------------------ anchors
class OnLoft:
    def __init__(self, loft, phi0, t0):
        self.loft, self.phi0, self.t0 = loft, phi0, t0
        self.r = float(loft.radius(phi0, t0)[0])

    def pn(self, s, t, off=0.0):
        return self.loft.pn(self.phi0 + np.asarray(s, float) / self.r, self.t0 + np.asarray(t, float), off)


class OnEllipsoid:
    def __init__(self, center, radii, n0, up=(0.0, 0.0, 1.0)):
        self.c, self.radii = np.asarray(center, float), np.asarray(radii, float)
        self.n0 = unit(np.asarray(n0, float))
        up = np.asarray(up, float)
        self.et = unit(up - (up @ self.n0) * self.n0)
        self.es = np.cross(self.et, self.n0)
        self.R = float(self.radii.mean())

    def pn(self, s, t, off=0.0):
        s, t = np.asarray(s, float).ravel(), np.asarray(t, float).ravel()
        rho = np.hypot(s, t) / self.R
        d = np.cos(rho)[:, None] * self.n0 + (np.sinc(rho / math.pi) / self.R)[:, None] * (s[:, None] * self.es + t[:, None] * self.et)
        N = unit(d / self.radii)
        off = np.asarray(off, float)
        return self.c + self.radii * d + (off.ravel()[:, None] if off.ndim else off) * N, N


class OnPlane:
    def __init__(self, origin, es, et):
        self.o, self.es, self.et = np.asarray(origin, float), unit(es), unit(et)
        self.n = np.cross(self.es, self.et)

    def pn(self, s, t, off=0.0):
        s, t = np.asarray(s, float).ravel(), np.asarray(t, float).ravel()
        off = np.asarray(off, float)
        N = np.tile(self.n, (len(s), 1))
        return self.o + s[:, None] * self.es + t[:, None] * self.et + (off.ravel()[:, None] if off.ndim else off) * N, N


def anchor_matrix(anchor, shift=(0.0, 0.0), lift=0.0):
    """4x4 right-handed frame at an anchor point: y along +t, z = outward normal, x = y cross z."""
    s0, t0 = shift
    h = 1e-3
    P, N = anchor.pn([s0], [t0], lift)
    et = unit(anchor.pn([s0], [t0 + h])[0][0] - anchor.pn([s0], [t0 - h])[0][0])
    n = N[0]
    et = unit(et - (et @ n) * n)
    es = np.cross(et, n)
    M = Matrix.Identity(4)
    for i in range(3):
        M[i][0], M[i][1], M[i][2], M[i][3] = es[i], et[i], n[i], P[0][i]
    return M


# ------------------------------------------------------------------------- pieces
def _finish(obj, thick=None, bevel=None, seg=2, subsurf=0, angle=42.0):
    if thick:
        m = obj.modifiers.new("Solidify", "SOLIDIFY")
        m.thickness, m.offset, m.use_even_offset, m.use_rim = thick, -1.0, True, True
    if bevel:
        m = obj.modifiers.new("Bevel", "BEVEL")
        m.width, m.segments, m.limit_method, m.angle_limit = bevel, seg, "ANGLE", math.radians(angle)
        m.harden_normals = True
    if subsurf:
        m = obj.modifiers.new("Subsurf", "SUBSURF")
        m.levels = m.render_levels = subsurf
    return obj


def plate(name, anchor, hs, ht, n=4.0, offset=0.0, thick=0.006, hole=0.0, dome=0.0, rot=0.0, shift=(0.0, 0.0),
          res=None, bevel=None, seg=2, subsurf=0, mat=None, coll="Marine"):
    """Shell plate with a superellipse outline, conforming to the anchor surface.

    `offset` is the height of the plate's TOP surface over the anchor surface;
    `thick` extends inwards. `hole` (0..1) leaves a concentric opening: a rim/frame.
    """
    res = res or RES
    m = int(np.clip(math.ceil(max(hs, ht) / res), 3, 40))
    ax = np.linspace(-1.0, 1.0, 2 * m + 1)
    A, B = np.meshgrid(ax, ax)
    rho = np.maximum(np.abs(A), np.abs(B))
    safe = np.maximum(rho, 1e-9)
    dx, dy = A / safe, B / safe
    k = np.where(rho > 0, np.maximum(np.abs(dx) ** n + np.abs(dy) ** n, 1e-12) ** (-1.0 / n), 0.0)  # concentric squares -> superellipses
    x, y = rho * k * dx * hs, rho * k * dy * ht
    cr, sr = math.cos(rot), math.sin(rot)
    s, t = x * cr - y * sr + shift[0], x * sr + y * cr + shift[1]
    P, N = anchor.pn(s.ravel(), t.ravel(), (offset + dome * (1.0 - rho**2)).ravel())
    side = 2 * m + 1
    idx = np.arange(side * side).reshape(side, side)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    f0 = faces[0]
    if np.cross(P[f0[1]] - P[f0[0]], P[f0[3]] - P[f0[0]]) @ N[f0[0]] < 0:  # (s, t, n) is left-handed on a loft
        faces = faces[:, ::-1]
    if hole > 0:
        ring = max(1, int(round(hole * m))) / m
        faces = faces[rho.ravel()[faces].max(axis=1) > ring + 1e-9]
    used = np.zeros(side * side, bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    obj = new_mesh(name, P[used], remap[faces], None, mat, coll)
    return _finish(obj, thick, bevel if bevel is not None else min(0.3 * thick, 0.003), seg, subsurf)


def box(name, size, M=None, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), bevel=0.004, seg=3, mat=None, coll="Marine", taper=None):
    """Bevelled box. `M` (4x4) or loc + XYZ-euler rot; `taper=(kx, ky)` scales the +z face."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        kx, ky = taper if (taper and v.co.z > 0) else (1.0, 1.0)
        v.co.x *= size[0] * kx
        v.co.y *= size[1] * ky
        v.co.z *= size[2]
    if M is None:
        M = Matrix.Translation(loc) @ Matrix.Rotation(rot[2], 4, "Z") @ Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
    bm.transform(M)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if mat is not None:
        me.materials.append(mat)
    obj = link(bpy.data.objects.new(name, me), coll)
    return _finish(obj, None, bevel, seg)


def box_at(name, anchor, size, shift=(0.0, 0.0), sink=0.0, **kw):
    """Box sitting on an anchor surface: size = (along s, along t, outward depth)."""
    M = anchor_matrix(anchor, shift, size[2] * 0.5 - sink)
    return box(name, size, M=M, **kw)


def tube(name, pts, normals=None, r=0.005, closed=False, n_u=10, mat=None, coll="Marine"):
    """Round tube swept along a polyline; `normals` (per point) fix the frame."""
    pts = np.asarray(pts, float)
    k = len(pts)
    if closed:
        T = unit(np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0))
    else:
        T = unit(np.gradient(pts, axis=0))
    if normals is None:  # parallel transport
        ref = np.array([0.0, 0.0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        normals = np.empty_like(pts)
        for i in range(k):
            ref = unit(ref - (ref @ T[i]) * T[i])
            normals[i] = ref
    Nn = unit(normals - (np.sum(normals * T, axis=1))[:, None] * T)
    Bn = np.cross(T, Nn)
    al = np.linspace(0.0, TAU, n_u, endpoint=False)
    V = pts[:, None, :] + r * (np.cos(al)[None, :, None] * Nn[:, None, :] + np.sin(al)[None, :, None] * Bn[:, None, :])
    idx = np.arange(k * n_u).reshape(k, n_u)
    nxt = np.roll(idx, -1, axis=1)
    if closed:
        up, upn = np.roll(idx, -1, axis=0), np.roll(nxt, -1, axis=0)
        faces = np.stack([idx, nxt, upn, up], axis=-1).reshape(-1, 4)
    else:
        faces = np.stack([idx[:-1], nxt[:-1], nxt[1:], idx[1:]], axis=-1).reshape(-1, 4)
    return new_mesh(name, V.reshape(-1, 3), faces, None, mat, coll)


def studs(name, pts, normals, r=0.004, flat=0.55, mat=None, coll="Marine"):
    """Rivet heads: one mesh of squashed hemispheres at pts, along normals."""
    pts, normals = np.asarray(pts, float).reshape(-1, 3), unit(np.asarray(normals, float).reshape(-1, 3))
    seg, rings = 10, 3
    el = np.linspace(0.0, 0.5 * math.pi, rings + 1)[:-1]
    al = np.linspace(0.0, TAU, seg, endpoint=False)
    loc = np.array([[math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), flat * math.sin(e)] for e in el for a in al] + [[0, 0, flat]]) * r
    V, Q, Tr = [], [], []
    for i, (p, n) in enumerate(zip(pts, normals)):
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = unit(np.cross(ref, n))
        e2 = np.cross(n, e1)
        base = i * len(loc)
        V.append(p + loc[:, :1] * e1 + loc[:, 1:2] * e2 + loc[:, 2:] * n)
        for a in range(rings - 1):
            for b in range(seg):
                Q.append([base + a * seg + b, base + a * seg + (b + 1) % seg, base + (a + 1) * seg + (b + 1) % seg, base + (a + 1) * seg + b])
        for b in range(seg):
            Tr.append([base + (rings - 1) * seg + b, base + (rings - 1) * seg + (b + 1) % seg, base + len(loc) - 1])
    return new_mesh(name, np.vstack(V), Q, Tr, mat, coll)


def band_on(name, loft, t0, t1, offset=0.004, thick=0.005, phi0=0.0, phi1=TAU, mat=None, coll="Marine", bevel=0.0015, **kw):
    """Strap / hem / belt: a strip of the loft surface lifted off it and given thickness."""
    obj = loft_mesh(name, loft, t0=t0, t1=t1, phi0=phi0, phi1=phi1, offset=offset, mat=mat, coll=coll, **kw)
    return _finish(obj, thick, bevel, 2)


def solid(obj, thick, bevel=0.002, seg=2):
    return _finish(obj, thick, bevel, seg)


def mirror(obj, merge=False):
    m = obj.modifiers.new("Mirror", "MIRROR")
    m.use_axis = (True, False, False)
    m.use_mirror_merge, m.merge_threshold, m.use_clip = merge, 2e-4, False
    if len(obj.modifiers) > 1:
        obj.modifiers.move(len(obj.modifiers) - 1, 0)
    return obj


# ----------------------------------------------------------------- faceted armour
def _facet_mesh(name, P, N, closed, thick, bevel, seg, angle, mat, coll):
    R, C = P.shape[:2]
    idx = np.arange(R * C).reshape(R, C)
    if closed:
        nxt = np.roll(idx, -1, axis=1)
        faces = np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], axis=-1).reshape(-1, 4)
    else:
        faces = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
    Pf, Nf = P.reshape(-1, 3), N.reshape(-1, 3)
    f0 = faces[0]
    if np.cross(Pf[f0[1]] - Pf[f0[0]], Pf[f0[3]] - Pf[f0[0]]) @ Nf[f0[0]] < 0:
        faces = faces[:, ::-1]
    return _finish(new_mesh(name, Pf, faces, None, mat, coll), thick, bevel, seg, angle=angle)


def facet_loft(name, loft, rows, thick=0.006, bevel=0.0015, seg=2, angle=14.0, closed=False, sag=True, mat=None, coll="Marine"):
    """Angular plate wrapped on a loft: a COARSE grid whose quads stay flat facets.

    `rows` = [(t, [(phi_deg, offset), ...]), ...] with the same count in every row, so a
    row may span a narrower arc than its neighbour (a pointed knee, a notched breastplate).
    Creases sharper than `angle` degrees are chamfered. A flat facet is a chord across a
    round limb; `sag` lifts every vertex by that chord's sagitta, so `offset` stays the
    clearance at the facet's MIDDLE and the limb never shows through the plate."""
    ph = np.radians([[p for p, _ in r[1]] for r in rows])
    off = np.array([[o for _, o in r[1]] for r in rows], float)
    tt = np.array([[r[0]] * ph.shape[1] for r in rows], float)
    if sag:
        d = np.abs(np.diff(ph, axis=1))
        gap = np.zeros_like(ph)
        gap[:, :-1] = d
        gap[:, 1:] = np.maximum(gap[:, 1:], d)
        off = off + loft.radius(ph.ravel(), tt.ravel()).reshape(ph.shape) * (1.0 - np.cos(0.5 * gap))
    P, N = loft.pn(ph.ravel(), tt.ravel(), off.ravel())
    return _facet_mesh(name, P.reshape(*ph.shape, 3), N.reshape(*ph.shape, 3), closed, thick, bevel, seg, angle, mat, coll)


def facet_anchor(name, anchor, rows, thick=0.006, bevel=0.0015, seg=2, angle=14.0, mat=None, coll="Marine"):
    """The same on any anchor: `rows` = [[(s, t, offset), ...], ...]."""
    A = np.array(rows, float)
    P, N = anchor.pn(A[..., 0].ravel(), A[..., 1].ravel(), A[..., 2].ravel())
    return _facet_mesh(name, P.reshape(*A.shape[:2], 3), N.reshape(*A.shape[:2], 3), False, thick, bevel, seg, angle, mat, coll)


def ribbon_on(name, loft, path, width, offset=0.003, thick=0.003, step=0.006, bevel=0.0008, mat=None, coll="Marine"):
    """Flat strap laid along a polyline `path` = [(phi_deg, t), ...] on a loft."""
    ph, tt = np.radians([p for p, _ in path]), np.array([t for _, t in path], float)
    r = float(loft.radius(ph[:1], tt[:1])[0])
    seg = np.hypot(np.diff(ph) * r, np.diff(tt))
    u = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0.0, u[-1], max(3, int(u[-1] / step) + 1))
    P, N = loft.pn(np.interp(q, u, ph), np.interp(q, u, tt), offset)
    T = unit(np.gradient(P, axis=0))
    side = unit(np.cross(N, T))
    V = np.concatenate([P - 0.5 * width * side, P + 0.5 * width * side])
    k = len(P)
    i = np.arange(k - 1)
    faces = np.stack([i, i + 1, i + 1 + k, i + k], axis=-1)
    if np.cross(V[1] - V[0], V[k] - V[0]) @ N[0] < 0:
        faces = faces[:, ::-1]
    return _finish(new_mesh(name, V, faces, None, mat, coll), thick, bevel, 2)


def tape_attr(g, paths):
    """`tape` attribute: distance to free polylines [(phi_deg, t), ...] on the loft, in the same
    encoding as `seam_attr`. For bands wider than the grid pitch (bonded seam tape, piping).
    A segment that runs straight along the limb (same phi at both ends) is snapped to a grid
    column, so its edges come out exactly straight."""
    d = np.full(g.phi.shape, 1.0)
    for path in paths:
        for (p1, t1), (p2, t2) in zip(path[:-1], path[1:]):
            p1, p2 = math.radians(p1), math.radians(p2)
            if p1 == p2:
                p1 = p2 = g.snap_phi(p1)
            ax, ay = wrap(g.phi - p1) * g.r, g.t - t1
            bx, by = wrap(p2 - p1) * g.r, t2 - t1
            k = np.clip((ax * bx + ay * by) / np.maximum(bx * bx + by * by, 1e-12), 0.0, 1.0)
            d = np.minimum(d, np.hypot(ax - k * bx, ay - k * by))
    return 1.0 - np.minimum(d, SEAM_CAP) / SEAM_CAP


def flip_x(obj):
    """Move a part built on the left limb to the right one (for asymmetric gear)."""
    obj.data.transform(Matrix.Scale(-1.0, 4, (1.0, 0.0, 0.0)))
    obj.data.flip_normals()
    return obj
