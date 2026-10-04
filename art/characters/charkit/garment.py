"""Garment pieces on the kit's surfaces: stitched patches (pockets, knee patches), panels of
any curved outline (`panel`: a bodysuit's mesh inserts), a shirt collar folded over a body
loft, zips, and a mesh orientation check.

Patches carry a `stitch` attribute (distance to a topstitch line inset from the edge), so
a material that reads it (wearmat.mat_coverall) draws a double row of stitching round
them. `attrs` callbacks take WORLD positions (n, 3), so one world-space field (dirt, say)
can be shared by every piece of a garment.
"""
import math

import bmesh
import bpy
import numpy as np

import kit
from kit import SEAM_CAP, _finish, new_mesh, unit, wrap


def _stitch_value(d):
    return 1.0 - np.minimum(np.abs(d), SEAM_CAP) / SEAM_CAP


def patch(name, anchor, hs, ht, offset=0.003, thick=0.003, n=8.0, shift=(0.0, 0.0), rot=0.0, dome=0.0, inset=0.0055,
          lines=(), bevel=None, seg=2, res=None, point=0.0, attrs=None, mat=None, coll="Garment"):
    """A sewn-on panel (pocket, flap, knee patch): `plate()`'s superellipse shell, plus a
    `stitch` attribute along a line `inset` in from the edge and along each local
    horizontal line in `lines` (t positions, e.g. a pocket's top hem). `offset` is the
    height of the panel's top surface over the anchor, `thick` its inward thickness.
    `point` draws the bottom edge down to a point at the middle (a chevron hem), by that much."""
    res = res or kit.RES
    m = int(np.clip(math.ceil(max(hs, ht) / (0.7 * res)), 4, 60))
    ax = np.linspace(-1.0, 1.0, 2 * m + 1)
    A, B = np.meshgrid(ax, ax)
    rho = np.maximum(np.abs(A), np.abs(B))
    safe = np.maximum(rho, 1e-9)
    dx, dy = A / safe, B / safe
    k = np.where(rho > 0, np.maximum(np.abs(dx) ** n + np.abs(dy) ** n, 1e-12) ** (-1.0 / n), 0.0)
    x, y = rho * k * dx * hs, rho * k * dy * ht
    edge = np.minimum(hs - np.abs(x), ht - np.abs(y)).ravel()
    if point:
        y = y - point * (1.0 - np.abs(x) / hs) * np.clip(-y / ht, 0.0, 1.0)
    cr, sr = math.cos(rot), math.sin(rot)
    s, t = x * cr - y * sr + shift[0], x * sr + y * cr + shift[1]
    P, N = anchor.pn(s.ravel(), t.ravel(), (offset + dome * (1.0 - rho ** 2)).ravel())
    d = np.abs(edge - inset)
    for ly in lines:
        d = np.minimum(d, np.where(np.abs(x.ravel()) < hs - inset * 0.5, np.abs(y.ravel() - ly), 1.0))
    side = 2 * m + 1
    idx = np.arange(side * side).reshape(side, side)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], axis=-1).reshape(-1, 4)
    f0 = faces[len(faces) // 2]
    if np.cross(P[f0[1]] - P[f0[0]], P[f0[3]] - P[f0[0]]) @ N[f0[0]] < 0:
        faces = faces[:, ::-1]
    va = dict(stitch=_stitch_value(d))
    for key, f in (attrs or {}).items():
        va[key] = np.asarray(f(P), float)
    obj = new_mesh(name, P, faces, None, mat, coll, attrs=va)
    return _finish(obj, thick, bevel if bevel is not None else min(0.4 * thick, 0.0025), seg)


def smooth_loop(pts, step):
    """A closed polyline [(u, v), ...] through its points (closed Catmull-Rom), resampled
    every `step` (same units). Returns (n, 2)."""
    P = np.asarray(pts, float)
    n = len(P)
    sub = 24
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        for s in np.linspace(0.0, 1.0, sub, endpoint=False):
            s2, s3 = s * s, s * s * s
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    Q = np.array(out)
    seg = np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    m = max(8, int(u[-1] / step))
    q = np.linspace(0.0, u[-1], m, endpoint=False)
    Qc = np.vstack([Q, Q[:1]])
    return np.column_stack([np.interp(q, u, Qc[:, 0]), np.interp(q, u, Qc[:, 1])])


def panel(name, loft, outline, offset=0.001, thick=0.001, res=None, lift=None, attrs=None, mat=None, coll="Garment", surface_uv=False):
    """An inset panel of any outline on a loft (a mesh insert, a shaped pad): `outline` =
    [(phi_rad, t), ...], a closed loop smoothed through its points. The panel is the loft's
    surface inside the loop, triangulated with the loop as its exact edge (no grid steps), lifted
    `offset` above the surface (plus `lift(P, N)`, metres along N, where the garment under it is
    displaced) and given `thick` inwards. Returns `(obj, rim)`: rim = the edge loop in world space
    on the panel's top surface, for piping. `surface_uv` stores the developed surface coordinates (metres
    round and along the loft) as the `mesh_u` / `mesh_v` attributes, for a pattern that lies IN
    the surface (wearmat.mat_mesh(surface=True)) instead of cutting it with world planes."""
    from mathutils import Vector
    from mathutils.geometry import delaunay_2d_cdt
    res = res or kit.RES
    O = np.asarray(outline, float)
    r0 = float(loft.radius([O[:, 0].mean()], [O[:, 1].mean()])[0])
    uv = np.column_stack([O[:, 0] * r0, O[:, 1]])
    B = smooth_loop(uv, 0.6 * res)
    # interior points on a grid, kept clear of the edge
    lo, hi = B.min(axis=0), B.max(axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0] + res * 0.5, hi[0], res), np.arange(lo[1] + res * 0.5, hi[1], res))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    inside = _inside(G, B) if len(G) else np.zeros(0, bool)
    G = G[inside]
    if len(G):
        dmin = np.min(np.linalg.norm(G[:, None, :] - B[None, :, :], axis=2), axis=1)
        G = G[dmin > 0.6 * res]
    V2 = np.vstack([B, G])
    nb = len(B)
    verts = [Vector((float(x), float(y))) for x, y in V2]
    edges = [(i, (i + 1) % nb) for i in range(nb)]
    out = delaunay_2d_cdt(verts, edges, [list(range(nb))], 1, 1e-7)
    V2o = np.array([[v[0], v[1]] for v in out[0]])
    tris = [f for f in out[2] if len(f) == 3]
    phi, t = V2o[:, 0] / r0, V2o[:, 1]
    P, N = loft.pn(phi, t)
    if lift is not None:
        P = P + np.asarray(lift(P, N), float)[:, None] * N
    P = P + offset * N
    T = np.array(tris, np.int64)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), N[T[:, 0]]) < 0
    T[flip] = T[flip][:, ::-1]
    va = {}
    for key, f in (attrs or {}).items():
        va[key] = np.asarray(f(P), float)
    if surface_uv:
        va["mesh_u"], va["mesh_v"] = V2o[:, 0], V2o[:, 1]
    obj = new_mesh(name, P, np.zeros((0, 4), np.int32), T, mat, coll, attrs=va)
    # the edge loop, in the order of B (CDT keeps the input verts first when it can; match by position)
    rim_idx = [int(np.argmin(np.linalg.norm(V2o - b, axis=1))) for b in B]
    return _finish(obj, thick, None, 2), P[rim_idx]


def _inside(Q, poly):
    """Even-odd point-in-polygon for points Q (n, 2) against a closed loop poly (m, 2)."""
    x, y = Q[:, 0][:, None], Q[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y0 > y) != (y1 > y)
    xc = x0 + (y - y0) * (x1 - x0) / np.where(y1 - y0 == 0, 1e-12, y1 - y0)
    return (cond & (x < xc)).sum(axis=1) % 2 == 1


def seg_dist(x, z, p0, p1):
    """Distance in the (x, z) plane from points to the segment p0 -> p1 (a stitched line
    seen from the front or back: a pocket opening, a dart)."""
    ax, az = x - p0[0], z - p0[1]
    bx, bz = p1[0] - p0[0], p1[1] - p0[1]
    k = np.clip((ax * bx + az * bz) / max(bx * bx + bz * bz, 1e-12), 0.0, 1.0)
    return np.hypot(ax - k * bx, az - k * bz)


def front_phi(loft, t, x=0.0):
    """Angle on the FRONT half of a half-body loft where its surface crosses world x
    (the centre front of a welded garment is x = 0 wherever the section's centre is)."""
    ph = np.linspace(0.0, math.pi, 1441)
    X = loft.pos(ph, np.full(ph.shape, t))[:, 0]
    i = np.argsort(X)
    return float(np.interp(x, X[i], ph[i]))


def shirt_collar(name, loft, t_top, gap_deg=26.0, stand=0.032, edge_drop=0.040, point_drop=0.050, point_span=40.0,
                 lift=0.004, hug=0.010, rows=26, cols=None, thick=0.003, neck=None, attrs=None, mat=None, coll="Garment"):
    """Half of a shirt collar on a HALF-BODY loft (mirror it with merge): one folded sheet
    that runs from the fall's edge on the shoulders, up the body `lift` above it to the
    neckline (`t_top`), over the fold `stand` higher, and back down inside as the stand,
    `hug` closer to the neck. The front is open by `gap_deg` either side of centre front;
    over the last `point_span` degrees the fall runs `point_drop` further down: the points.
    Its section angle runs from the back (-90 deg) round the side to the front gap.

    `neck` = (cy, a, bf, bb, clear) hugs the collar to the neck: the fold sits `clear` off
    a level ellipse round the neck (centre y `cy`, half-width `a`, half-depths to the front
    and back) instead of straight above the neckline, and the stand runs down the neck
    inside it. None keeps the stand standing straight up off the neckline."""
    phis = np.radians(np.linspace(-90.0, 90.0 - gap_deg, cols or 72))
    c_top = loft.frames([t_top])[0][0]
    z_top = float(loft.pos([0.0], [t_top])[0][2])
    path_rows = []
    for ph in phis:
        to_gap = math.degrees(math.radians(90.0 - gap_deg) - ph)
        w = np.clip(1.0 - to_gap / point_span, 0.0, 1.0) ** 2
        z_edge = z_top - edge_drop - point_drop * w
        t_edge = _t_at_z_on(loft, ph, z_edge, t_top)
        ts = np.linspace(t_edge, t_top, 8)
        P, N = loft.pn(np.full(8, ph), ts, lift + 0.002 * (ts - t_edge) / max(t_top - t_edge, 1e-6))
        top = P[-1]
        rad = top - c_top
        rad[2] = 0.0
        rad = unit(rad)
        if neck is None:
            fold = top + np.array([0.0, 0.0, stand * (1.0 - 0.35 * w)]) + rad * 0.003
            inner_top = fold - rad * (hug + 0.004) - np.array([0.0, 0.0, 0.006])
            inner_bot = top - rad * (hug + 0.002) - np.array([0.0, 0.0, 0.016])
        else:
            cy, na, nbf, nbb, clear = neck
            v = np.array([top[0], top[1] - cy])
            u = v / max(np.linalg.norm(v), 1e-9)
            nb = nbf if u[1] < 0 else nbb
            rn = 1.0 / math.sqrt((u[0] / na) ** 2 + (u[1] / nb) ** 2)
            ring = lambda r, z: np.array([u[0] * r, cy + u[1] * r, z])
            # never outside the neckline itself (where the neckline is tight the stand rises
            # straight), or the sheet would fold back over itself
            rt = float(np.linalg.norm(v)) - 0.001
            # the stand's inner face runs 7 mm inside the fold, more than the cloth's thickness,
            # so the solidified sheet never meets itself at the turn
            ro = min(rn + clear + 0.003, rt)
            fold = ring(ro, top[2] + stand * (1.0 - 0.35 * w))
            inner_top = ring(ro - 0.007, top[2] + stand * (1.0 - 0.35 * w) - 0.007)
            inner_bot = ring(ro - 0.008, top[2] - 0.016)
        pts = np.vstack([P, fold, inner_top, inner_bot])
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        q = np.linspace(0.0, u[-1], rows)
        path_rows.append(np.column_stack([np.interp(q, u, pts[:, k]) for k in range(3)]))
    V = np.stack(path_rows, axis=1)  # (rows, cols, 3)
    R, C = V.shape[:2]
    idx = np.arange(R * C).reshape(R, C)
    faces = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
    Vf = V.reshape(-1, 3)
    va = {}
    for key, f in (attrs or {}).items():
        va[key] = np.asarray(f(Vf), float)
    # topstitching 5 mm in from the fall's edge
    edge_d = np.zeros((R, C))
    seg = np.linalg.norm(np.diff(V, axis=0), axis=2)
    edge_d[1:] = np.cumsum(seg, axis=0)
    va["stitch"] = _stitch_value(edge_d - 0.005).ravel()
    obj = new_mesh(name, Vf, faces, None, mat, coll, attrs=va)
    return _finish(obj, thick, 0.0012, 2)


def _t_at_z_on(loft, phi, z, t_hi):
    ts = np.linspace(0.0, t_hi, 400)
    zs = loft.pos(np.full(ts.shape, phi), ts)[:, 2]
    return float(np.interp(z, zs, ts))


# --------------------------------------------------------------------- mesh checks
def _evaluated(ob, dg):
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", T)
    T = T.reshape(-1, 3)
    bm = bmesh.new()
    bm.from_mesh(me)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    ev.to_mesh_clear()
    return V, T, boundary


def close_holes(objects):
    """Cap every open boundary loop of each object's SOURCE mesh with a fan (an open
    finger base, a tube's end, a stud's underside): parts become closed solids."""
    for ob in objects:
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        edges = [e for e in bm.edges if e.is_boundary]
        if edges:
            res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
            bmesh.ops.triangulate(bm, faces=res["faces"])
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(ob.data)
            ob.data.update()
        bm.free()
    return objects


def orient_outward(objects=None, ratio=0.05, verbose=True):
    """Make every mesh's normals face OUT: the signed volume of its evaluated mesh (after
    mirror and solidify) must be positive; a mesh that comes out inside-out has its SOURCE
    normals flipped (so its solidify grows inwards again, as the kit intends). Open sheets
    whose signed volume is too small to tell (|vol| < ratio * total) are left alone and
    listed. Returns a report dict: flipped, open, closed counts."""
    dg = bpy.context.evaluated_depsgraph_get()
    flipped, undecided, closed, opened = [], [], 0, []
    for ob in list(objects or bpy.data.objects):
        if ob.type != "MESH" or ob.name == "Floor":
            continue
        V, T, boundary = _evaluated(ob, dg)
        if len(T) == 0:
            continue
        c = V.mean(axis=0)
        a, b, d = V[T[:, 0]] - c, V[T[:, 1]] - c, V[T[:, 2]] - c
        v6 = np.einsum("ij,ij->i", a, np.cross(b, d))
        tot, absv = v6.sum(), np.abs(v6).sum()
        if boundary:
            opened.append(ob.name)
        else:
            closed += 1
        # a closed mesh's signed volume is exact whatever its shape (a thin shell has a small but
        # positive one); an open sheet's is only meaningful when it is a large share of the total
        if absv <= 0 or (boundary and abs(tot) < ratio * absv):
            undecided.append(ob.name)
        elif tot < 0:
            ob.data.flip_normals()
            flipped.append(ob.name)
    if flipped:
        bpy.context.view_layer.update()
    if verbose:
        print("normals: flipped %d %s; undecided %s; closed %d, open %d %s" % (len(flipped), flipped, undecided, closed, len(opened), opened))
    return dict(flipped=flipped, undecided=undecided, closed=closed, open=opened)
