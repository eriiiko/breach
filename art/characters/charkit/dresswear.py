"""Dress-uniform pieces on the kit (Blender 4.5, numpy + bpy): free-edged regions of a loft,
straps laid across a welded half-body, tapered tubes, riding boots and spurs.

Written for the first French Imperial Guard officer (`art/characters/french-imperial-guard`)
and meant for any character in tailored or period dress:

  loft_region   a patch of a loft between edges that vary along it: a boot shaft cut to a V
                at the top, a pointed cuff, a pelisse whose hem and front edges are curves
  half_body_pn  points and normals on a HALF-body loft (welded at x = 0, `workwear.Figure`)
                at signed world x, so a strap can cross the centre front
  strap         a flat strap through such points (a cross-belt)
  swept         a tube of varying radius and flattening along a polyline, capped: cords,
                a moustache, a chin chain, an empty sleeve
  riding_boot   a tall polished boot: foot, sole, low heel, a shaft to under the knee whose top
                edge is a table (`top`: back, side, front points and a front notch)
  spur          strap round the ankle, neck and a rowel

Conventions as the kit: faces -Y, +X is the figure's LEFT, limbs on the left and mirrored.
"""
import math

import numpy as np
from mathutils import Matrix

import kit
from kit import TAU, Loft, _finish, mirror, new_mesh, unit

Z = np.array([0.0, 0.0, 1.0])
D = math.radians


# --------------------------------------------------------------------- grid meshes
def grid_faces(R, C, closed=False):
    """Quads over an (R, C) vertex grid; `closed` wraps the columns."""
    idx = np.arange(R * C).reshape(R, C)
    if closed:
        nxt = np.roll(idx, -1, axis=1)
        return np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], axis=-1).reshape(-1, 4)
    return np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)


def orient(faces, P, N):
    """Reverse the winding if the middle face's normal points against N there."""
    f0 = faces[len(faces) // 2]
    n = np.cross(P[f0[1]] - P[f0[0]], P[f0[3]] - P[f0[0]])
    return faces[:, ::-1] if n @ N[f0[0]] < 0 else faces


def loft_region(name, loft, phi_of, t_of, res=None, closed=False, offset=0.0, disp=None, attrs=None, mat=None, coll="Garment",
                rows=None, cols=None):
    """A patch of `loft` between free edges: `phi_of(u, v)` and `t_of(u, v)` map the unit square
    (u along the loft, v around it) onto section angle and arclength, so each edge may be any
    curve. `closed` wraps v round the whole section (v in [0, 1) then). The grid is sized to `res`
    from the patch's own extent unless `rows` / `cols` are given. `offset`, `disp` and `attrs`
    callbacks see a `kit.Grid` (phi, t, r, P, N, u, v), as on `kit.loft_mesh`."""
    res = res or kit.RES
    if rows is None or cols is None:
        uu, vv = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9), indexing="ij")
        Ps = loft.pos(np.asarray(phi_of(uu, vv), float).ravel(), np.clip(np.asarray(t_of(uu, vv), float).ravel(), 0, loft.L)).reshape(9, 9, 3)
        len_u = np.linalg.norm(np.diff(Ps, axis=0), axis=2).sum(axis=0).max()
        len_v = np.linalg.norm(np.diff(Ps, axis=1), axis=2).sum(axis=1).max()
        rows = rows or max(2, int(round(len_u / res)) + 1)
        cols = cols or max(4, int(round(len_v / res)) + 1)
    u = np.linspace(0.0, 1.0, rows)
    v = np.linspace(0.0, 1.0, cols, endpoint=not closed)
    U, V = np.meshgrid(u, v, indexing="ij")
    PH = np.asarray(phi_of(U, V), float) * np.ones_like(U)
    TT = np.clip(np.asarray(t_of(U, V), float) * np.ones_like(U), 0.0, loft.L)
    g = kit.Grid()
    g.loft, g.phi, g.t, g.u, g.v, g.res = loft, PH, TT, U, V, res
    g.rows, g.cols = TT[:, 0], PH[0]
    g.r = loft.radius(PH.ravel(), TT.ravel()).reshape(PH.shape)
    off = offset(g) if callable(offset) else offset
    P, N = loft.pn(PH.ravel(), TT.ravel(), np.asarray(off, float).ravel() if np.ndim(off) else off)
    g.P, g.N = P.reshape(*PH.shape, 3), N.reshape(*PH.shape, 3)
    if disp is not None:
        P = P + np.asarray(disp(g), float).ravel()[:, None] * N
    va = {k: np.asarray(f(g), float).ravel() for k, f in (attrs or {}).items()}
    faces = orient(grid_faces(rows, cols, closed), P, N)
    return new_mesh(name, P, faces, None, mat, coll, va)


# ------------------------------------------------------------- the welded half-body
def half_body_pn(loft, x, z, back=False, offset=0.0, n_phi=721):
    """Points and outward normals on a half-body loft (x >= 0, welded to its mirror at x = 0)
    at signed world x and height z, on the front (or back) of the body. Rows of the loft must
    be level there (the torso, `ring(..., pin=1)`)."""
    x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
    ts = np.linspace(0.0, loft.L, 900)
    t = np.interp(z, loft.frames(ts)[0][:, 2], ts)
    lo, hi = (math.pi, TAU) if back else (0.0, math.pi)
    ph = np.linspace(lo, hi, n_phi)
    phis = np.empty(len(x))
    for i in range(len(x)):
        X = loft.pos(ph, np.full(n_phi, t[i]))[:, 0]
        o = np.argsort(X)
        phis[i] = np.interp(abs(x[i]), X[o], ph[o])
    P, N = loft.pn(phis, t, offset)
    neg = x < 0
    P[neg, 0] *= -1.0
    N[neg, 0] *= -1.0
    return P, N


def strap(name, P, N, width, thick=0.003, bevel=0.0008, attrs=None, mat=None, coll="Garment"):
    """A flat strap through points P with surface normals N (from `half_body_pn` with the lift in
    its offset), `width` across, `thick` inwards. `attrs` = {name: f(u)} along it (0..1)."""
    P, N = np.asarray(P, float), unit(np.asarray(N, float))
    T = unit(np.gradient(P, axis=0))
    side = unit(np.cross(N, T))
    V = np.concatenate([P - 0.5 * width * side, P + 0.5 * width * side])
    k = len(P)
    i = np.arange(k - 1)
    faces = np.stack([i, i + 1, i + 1 + k, i + k], axis=-1)
    if np.cross(V[1] - V[0], V[k] - V[0]) @ N[0] < 0:
        faces = faces[:, ::-1]
    u = np.linspace(0.0, 1.0, k)
    va = {key: np.tile(np.asarray(f(u), float), 2) for key, f in (attrs or {}).items()}
    return _finish(new_mesh(name, V, faces, None, mat, coll, va), thick, bevel, 2)


def smooth_path(pts, n=40):
    """A Catmull-Rom curve through control points, resampled to n points by arclength."""
    pts = np.asarray(pts, float)
    tk = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    q = np.linspace(0.0, tk[-1], n * 4)
    c = kit.spline(tk, pts, q)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    return np.column_stack([np.interp(np.linspace(0, s[-1], n), s, c[:, k]) for k in range(3)])


def swept(name, pts, radii, n_u=12, flat=1.0, up=None, caps=True, attrs=None, disp=None, mat=None, coll="Garment"):
    """A capped tube along the polyline `pts` whose section radius is `radii` (scalar or one per
    point); `flat` (scalar or per point) squashes the section along the frame's second axis.
    The frame keeps its first axis towards `up` (a vector or one per point) if given, else it is
    parallel-transported. `attrs` = {name: f(u, al)} over the length u (0..1) and the angle al,
    `disp(u, al)` (m) pushes the surface out radially. Closed: both ends are fanned shut."""
    pts = np.asarray(pts, float)
    k = len(pts)
    r = np.broadcast_to(np.asarray(radii, float), (k,)).astype(float)
    fl = np.broadcast_to(np.asarray(flat, float), (k,)).astype(float)
    T = unit(np.gradient(pts, axis=0))
    if up is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        Nn = np.empty_like(pts)
        for i in range(k):
            ref = unit(ref - (ref @ T[i]) * T[i])
            Nn[i] = ref
    else:
        upv = np.broadcast_to(np.asarray(up, float), (k, 3))
        Nn = unit(upv - np.sum(upv * T, axis=1)[:, None] * T)
    Bn = np.cross(T, Nn)
    al = np.linspace(0.0, TAU, n_u, endpoint=False)
    u = np.linspace(0.0, 1.0, k)
    UU, AL = np.meshgrid(u, al, indexing="ij")
    rr = r[:, None] + (np.asarray(disp(UU, AL), float) if disp is not None else 0.0)
    V = pts[:, None, :] + rr[..., None] * (np.cos(al)[None, :, None] * Nn[:, None, :] + (fl[:, None] * np.sin(al)[None, :])[..., None] * Bn[:, None, :])
    V = V.reshape(-1, 3)
    idx = np.arange(k * n_u).reshape(k, n_u)
    nxt = np.roll(idx, -1, axis=1)
    quads = np.stack([idx[:-1], nxt[:-1], nxt[1:], idx[1:]], axis=-1).reshape(-1, 4)
    va = {key: np.asarray(f(UU, AL), float).ravel() for key, f in (attrs or {}).items()}
    tris = None
    if caps:
        c0, c1 = len(V), len(V) + 1
        V = np.vstack([V, pts[0], pts[-1]])
        tris = np.vstack([np.stack([np.full(n_u, c0), nxt[0], idx[0]], axis=-1), np.stack([np.full(n_u, c1), idx[-1], nxt[-1]], axis=-1)])
        va = {key: np.append(a, [a[:n_u].mean(), a[-n_u:].mean()]) for key, a in va.items()}
    # outward winding: the first quad's normal against the radial direction
    q = quads[len(quads) // 2]
    nq = np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]])
    i0 = q[0] // n_u
    if nq @ (V[q[0]] - pts[i0]) < 0:
        quads = quads[:, ::-1]
        if tris is not None:
            tris = tris[:, ::-1]
    return new_mesh(name, V, quads, tris, mat, coll, va)


# ---------------------------------------------------------------------- riding boot
class RidingBoot:
    """A tall riding boot from `spec` (see `riding_boot`), on the LEFT leg."""

    def __init__(self, spec):
        self.s = s = spec
        c, sn = math.cos(D(s["toe_out"])), math.sin(D(s["toe_out"]))
        self.BX, self.BY = np.array([c, sn, 0.0]), np.array([-sn, c, 0.0])
        self.ankle = np.array([s["ankle"][0], s["ankle"][1], 0.0])
        prof = s["profile"]
        self.y_toe, self.y_heel = prof[0][0], prof[-1][0]

    def w(self, x, y, z):
        return self.ankle + x * self.BX + y * self.BY + z * Z

    def sole_top(self, y):
        """Top of the sole along the foot: `sole` thick from the toe to the ball, rising to the
        heel's height behind the heel's front edge."""
        s = self.s
        y_ball, y_heel_front = s["ball_y"], s["heel_front_y"]
        k = np.clip((np.asarray(y, float) - y_ball) / (y_heel_front - y_ball), 0.0, 1.0)
        return s["sole"] + (s["heel"] - s["sole"]) * k * k * (3 - 2 * k)

    def top_z(self, phi):
        """Height of the shaft's top edge round the leg (phi 90 deg = front, 270 = back)."""
        tp = self.s["top"]
        f = np.sin(phi)  # 1 front, -1 back
        z = np.where(f >= 0, tp["side"] + (tp["front"] - tp["side"]) * np.maximum(f, 0.0) ** 0.7, tp["side"] + (tp["back"] - tp["side"]) * (-f))
        d = np.abs(kit.wrap(phi - D(90.0))) / D(tp["notch_w"])
        return z - (tp["front"] - tp["notch"]) * np.clip(1.0 - d, 0.0, 1.0) ** 1.4


def riding_boot(prefix, spec, mats, coll="Boots", attrs=None):
    """A polished riding boot on the LEFT leg, mirrored: a foot (one loft toe -> heel on a sole
    and a low heel block), and a shaft (a loft of level rings ankle -> top) whose top edge is
    the `top` table (`RidingBoot.top_z`); a lace TRIM band along that edge. Returns the boot.

    spec keys: ankle (x, y), toe_out deg, sole / heel (m), ball_y, heel_front_y, profile ((y
    along the foot from the ankle, half-width, height of the upper), ...) toe -> heel, shaft
    ((z, cx, cy, a, bf, bb), ...), top dict(back, side, front, notch, notch_w deg), trim (m).
    mats: leather, sole, trim."""
    bt = RidingBoot(spec)
    s, w, st = bt.s, bt.w, bt.sole_top
    prof = s["profile"]
    leather = mats["leather"]
    objs = []
    # the sole: a slab under the outline, the heel block under the heel
    sole = Loft([dict(p=w(0, y, 0.5 * s["sole"]), a=a + 0.003, bf=0.5 * s["sole"], bb=0.5 * s["sole"], n=7.0) for y, a, _ in prof], front=Z)
    objs.append(mirror(kit.loft_mesh(prefix + "_Sole", sole, res=0.003, cap0=True, cap1=True, mat=mats["sole"], coll=coll)))
    hy0, hy1 = s["heel_front_y"], prof[-1][0]
    heel_rows = [(y, a) for y, a, _ in prof if y >= hy0 - 1e-9]
    hz = 0.5 * s["heel"]
    heel = Loft([dict(p=w(0, y, hz), a=a + 0.001, bf=hz, bb=hz, n=8.0) for y, a in heel_rows], front=Z)
    objs.append(mirror(kit.loft_mesh(prefix + "_Heel", heel, res=0.003, cap0=True, cap1=True, mat=mats["sole"], coll=coll)))
    foot = Loft([dict(p=w(0, y, float(st(y)) + 0.006), a=a, bf=h, bb=0.008, n=(2.4, 6.0)) for y, a, h in prof], front=Z)
    objs.append(mirror(kit.loft_mesh(prefix + "_Foot", foot, res=0.003, cap0=True, cap1=True, mat=leather, coll=coll,
                                     attrs=attrs and {k: (lambda g, f=f: f(g.P)) for k, f in attrs.items()})))
    # the shaft, cut to the top edge
    shaft = Loft([dict(p=(cx, cy, z), a=a, bf=bf, bb=bb, n=2.1, t=(0.0, 0.0, 1.0)) for z, cx, cy, a, bf, bb in s["shaft"]])
    ts = np.linspace(0.0, shaft.L, 400)
    zs = shaft.frames(ts)[0][:, 2]
    t_of_z = lambda z: np.interp(z, zs, ts)
    top_t = lambda phi: t_of_z(bt.top_z(phi))
    phi_of = lambda u, v: TAU * v
    objs.append(mirror(kit.solid(loft_region(prefix + "_Shaft", shaft, phi_of, lambda u, v: u * top_t(TAU * v), closed=True, mat=leather, coll=coll,
                                             attrs=attrs and {k: (lambda g, f=f: f(g.P)) for k, f in attrs.items()}), 0.003, bevel=0.0)))
    tw = s["trim"]
    trim = loft_region(prefix + "_Trim", shaft, phi_of, lambda u, v: top_t(TAU * v) - tw * (1.0 - u) * 1.0, closed=True, offset=0.0022,
                       mat=mats["trim"], coll=coll, res=0.0025)
    objs.append(mirror(kit.solid(trim, 0.0035, bevel=0.0)))
    bt.shaft, bt.top_t = shaft, top_t
    return bt, objs


def spur(prefix, bt, spec, mat, coll="Boots"):
    """A spur on a `RidingBoot`'s heel, mirrored: a strap round the ankle (a closed tube in a
    plane tilted from the heel up over the instep), the neck backwards from the heel and a
    star rowel turning about x. spec: strap (z at the heel, z at the instep), r, neck (length,
    rise), rowel (radius, points)."""
    s = bt.s
    ring_z0, ring_z1 = spec["strap"]
    al = np.linspace(0.0, TAU, 48, endpoint=False)
    sh = bt.shaft
    pts, objs = [], []
    for a in al:  # a = 0 the heel, pi the instep
        z = ring_z0 + (ring_z1 - ring_z0) * 0.5 * (1.0 - math.cos(a))
        t = float(np.interp(z, sh.frames(np.linspace(0, sh.L, 200))[0][:, 2], np.linspace(0, sh.L, 200)))
        phi = D(270.0) + a  # from the back round the outside to the front
        P, _ = sh.pn([phi], [t], spec["r"] + 0.0015)
        pts.append(P[0])
    objs.append(mirror(kit.tube(prefix + "_Spur_Strap", np.array(pts), r=spec["r"], closed=True, n_u=8, mat=mat, coll=coll)))
    t0 = float(np.interp(ring_z0, sh.frames(np.linspace(0, sh.L, 200))[0][:, 2], np.linspace(0, sh.L, 200)))
    heel_p, heel_n = sh.pn([D(270.0)], [t0], 0.002)
    length, rise = spec["neck"]
    back = unit(np.array([0.0, heel_n[0][1], 0.0]))
    tip = heel_p[0] + back * length + Z * rise
    neck = smooth_path([heel_p[0], heel_p[0] + back * 0.5 * length + Z * 0.3 * rise, tip], 12)
    objs.append(mirror(swept(prefix + "_Spur_Neck", neck, np.linspace(0.0055, 0.0035, 12), n_u=10, mat=mat, coll=coll)))
    rr, k = spec["rowel"]
    # the rowel: a star polygon in the plane holding the neck (normal = x), as a thin prism
    ang = np.linspace(0.0, TAU, 2 * k, endpoint=False)
    rad = np.where(np.arange(2 * k) % 2 == 0, rr, rr * 0.35)
    e1, e2 = back, Z
    ring = tip[None, :] + rad[:, None] * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
    hx = np.array([0.0012, 0.0, 0.0])
    V = np.vstack([ring - hx, ring + hx, tip - hx, tip + hx])
    n = 2 * k
    q = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    tr = [[2 * n, (i + 1) % n, i] for i in range(n)] + [[2 * n + 1, n + i, n + (i + 1) % n] for i in range(n)]
    objs.append(mirror(new_mesh(prefix + "_Spur_Rowel", V, q, tr, mat, coll)))
    return objs


def flip_built(obj):
    """Move a part built on the figure's LEFT to the right (kit.flip_x, kept here for symmetry
    of naming with the uniform's one-sided pieces)."""
    return kit.flip_x(obj)


def matrix_at(P, x_axis, z_axis):
    """4x4 frame at P with its x along `x_axis` and z along `z_axis` (made orthogonal)."""
    zz = unit(np.asarray(z_axis, float))
    xx = unit(np.asarray(x_axis, float) - (np.asarray(x_axis, float) @ zz) * zz)
    yy = np.cross(zz, xx)
    M = Matrix.Identity(4)
    for i in range(3):
        M[i][0], M[i][1], M[i][2], M[i][3] = xx[i], yy[i], zz[i], P[i]
    return M
