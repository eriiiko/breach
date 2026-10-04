"""Garment pieces on the kit's surfaces: stitched patches (pockets, knee patches), a shirt
collar folded over a body loft, zips, and a mesh orientation check.

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
          lines=(), bevel=None, seg=2, res=None, attrs=None, mat=None, coll="Garment"):
    """A sewn-on panel (pocket, flap, knee patch): `plate()`'s superellipse shell, plus a
    `stitch` attribute along a line `inset` in from the edge and along each local
    horizontal line in `lines` (t positions, e.g. a pocket's top hem). `offset` is the
    height of the panel's top surface over the anchor, `thick` its inward thickness."""
    res = res or kit.RES
    m = int(np.clip(math.ceil(max(hs, ht) / (0.7 * res)), 4, 60))
    ax = np.linspace(-1.0, 1.0, 2 * m + 1)
    A, B = np.meshgrid(ax, ax)
    rho = np.maximum(np.abs(A), np.abs(B))
    safe = np.maximum(rho, 1e-9)
    dx, dy = A / safe, B / safe
    k = np.where(rho > 0, np.maximum(np.abs(dx) ** n + np.abs(dy) ** n, 1e-12) ** (-1.0 / n), 0.0)
    x, y = rho * k * dx * hs, rho * k * dy * ht
    cr, sr = math.cos(rot), math.sin(rot)
    s, t = x * cr - y * sr + shift[0], x * sr + y * cr + shift[1]
    P, N = anchor.pn(s.ravel(), t.ravel(), (offset + dome * (1.0 - rho ** 2)).ravel())
    edge = np.minimum(hs - np.abs(x), ht - np.abs(y)).ravel()
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
            fold = ring(rn + clear + 0.003, top[2] + stand * (1.0 - 0.35 * w))
            inner_top = ring(rn + clear, top[2] + stand * (1.0 - 0.35 * w) - 0.006)
            inner_bot = ring(rn + clear * 0.6, top[2] - 0.016)
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
