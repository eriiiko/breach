"""A hair BLOCKOUT for `animehead` heads (Beatrice B2a): a cap over the skull and smooth locks with the drawing's main
partings and silhouette. Strand detail is a later round.

  cap    the egg (`animehead.Egg`, head space) offset outwards by a thickness above the hairline and taken deep
         inside the head below it, so the visible hairline is a clean intersection with the skin; meshed like the skin
  locks  each a closed swept ribbon (lens section, tapering to a point): an ON-HEAD part laid on the cap along a path in
         the head's angles (beta = elevation, alpha = azimuth from the front towards her left, about the egg's centre),
         then (if it is long) a HANGING part that falls from where it leaves the head as a rope under gravity, kept
         `clear` off the head and off the suit (a BVH of the suit's objects) by position-based relaxation. The ribbon's
         flat side faces away from what it lies on.

Groups are table rows (`HAIR` in the character's tables); each row makes `n` locks spread across a range, mirrored.
Every lock gets `hu` (along, m) and `hv` (across, m, offset per lock) for the material's streaks.
"""
import math

import numpy as np

import animehead as ah


def direction(beta, alpha):
    b, a = np.radians(beta), np.radians(alpha)
    return np.stack([np.cos(b) * np.sin(a), -np.cos(b) * np.cos(a), np.sin(b)], axis=-1)


class Hair:
    def __init__(self, head, spec):
        self.h, self.sp = head, spec
        self.egg = head.egg
        cap = spec["cap"]
        lp = np.array([p for p, _ in cap["line"]], float)
        lz = np.array([z for _, z in cap["line"]], float)
        o = np.argsort(lp)
        self.lp = np.concatenate([lp[o] - 360.0, lp[o], lp[o] + 360.0])
        self.lz = np.concatenate([lz[o], lz[o], lz[o]])

    # --- the cap (head space)
    def line_z(self, alpha):
        return np.interp(np.mod(alpha, 360.0), self.lp, self.lz)

    def thick(self, Q):
        """Cap thickness at head-space points: `thick` (front, crown, back) by direction."""
        c = self.egg.c
        D = ah.unit(Q - c)
        tf, tc, tb = self.sp["cap"]["thick"]
        up = np.clip(D[:, 2], 0.0, 1.0)
        fb = np.clip(-D[:, 1] / np.maximum(np.hypot(D[:, 0], D[:, 1]), 1e-9), -1.0, 1.0)
        side = tf * np.clip(fb, 0.0, 1.0) + tb * np.clip(-fb, 0.0, 1.0) + 0.5 * (tf + tb) * (1.0 - np.abs(fb))
        return side + (tc - side) * up ** 2

    def cover(self, Q):
        c = self.egg.c
        alpha = np.degrees(np.arctan2(Q[:, 0], -(Q[:, 1] - c[1])))   # 0 the front, 90 her left
        return ah._ramp(Q[:, 2] - self.line_z(np.abs(alpha)), 0.0, self.sp["cap"]["fade"])

    def cap_field(self, P):
        Q = self.h.to_head(P)
        m = self.cover(Q)
        T = self.thick(Q) * m - self.sp["cap"]["under"] * (1.0 - m)
        return (self.egg(Q) - T) * self.h.s

    # --- lock paths
    def on_head(self, beta, alpha, off):
        """World points on the cap: along directions (beta, alpha) at the egg's radius + cap thickness + off."""
        D = direction(beta, alpha)
        c = self.egg.c
        r = self.egg.rho(D)
        Q = c + D * r[:, None]
        Q = c + D * (r + self.thick(Q) * self.cover(Q) + off)[:, None]
        return self.h.world(Q)

    def hang(self, P0, T0, length, n, clear, body, iters=80, pull=None):
        """A rope of n points from P0 leaving along T0, `length` long, under gravity, kept `clear` (head, body) off what
        it falls onto. `pull`: an optional point the rope's middle leans towards (a lock brought over the shoulder)."""
        seg = length / (n - 1)
        T0 = ah.unit(T0)
        s = np.arange(n) * seg
        down = np.array([0.0, 0.0, -1.0])
        bend = np.clip(s / 0.06, 0.0, 1.0)[:, None]
        dirs = ah.unit(T0[None, :] * (1.0 - bend) + down[None, :] * bend)
        P = P0 + np.concatenate([[np.zeros(3)], np.cumsum(dirs[:-1] * seg, axis=0)])
        if pull is not None:
            w = np.sin(np.pi * np.clip(s / length, 0, 1)) ** 2
            P += (np.asarray(pull, float) - P) * 0.3 * w[:, None] * np.clip(s / 0.05, 0, 1)[:, None]
        g = np.array([0.0, 0.0, -0.25 * seg])
        for it in range(iters):
            P[2:] += g
            if pull is not None:
                P[2:] += (np.asarray(pull, float) - P[2:]) * 0.01 * np.sin(np.pi * s[2:] / length)[:, None]
            for _ in range(3):
                d = P[1:] - P[:-1]
                L = np.linalg.norm(d, axis=1, keepdims=True)
                corr = d * (1.0 - seg / np.maximum(L, 1e-9)) * 0.5
                P[1:] -= corr
                P[:-1] += corr
                P[0] = P0
                P[1] = P0 + T0 * seg
                # bending: neighbours two apart no closer than 1.9 segments
                d2 = P[2:] - P[:-2]
                L2 = np.linalg.norm(d2, axis=1, keepdims=True)
                c2 = d2 * np.minimum(0.0, 1.0 - 1.9 * seg / np.maximum(L2, 1e-9)) * 0.25
                P[2:] -= c2
                P[:-2] += c2
                P[0] = P0
                P[1] = P0 + T0 * seg
            P[2:] = self.push(P[2:], clear, body)
        return P

    def push(self, P, clear, body):
        f = self.cap_field(P)
        bad = f < clear
        if bad.any():
            g = ah.unit(self.h.grad(self.cap_field, P[bad]))
            P[bad] -= (f[bad] - clear)[:, None] * g
        if body is not None:
            for i in range(len(P)):
                co, nrm, _, dist = body.find_nearest(tuple(P[i]))
                if co is None:
                    continue
                co, nrm = np.array(co), np.array(nrm)
                side = (P[i] - co) @ nrm
                if side < clear:
                    P[i] = co + nrm * clear + (P[i] - co - nrm * side)
        return P


def _smooth(P, k=3, fix=2):
    P = P.copy()
    for _ in range(k):
        P[fix:-1] = 0.25 * P[fix - 1:-2] + 0.5 * P[fix:-1] + 0.25 * P[fix + 1:]
    return P


def _resample(P, m):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], m)
    return np.column_stack([np.interp(t, s, P[:, i]) for i in range(3)]), s[-1]


def lock_paths(hair, row, body, rng):
    """The paths of one group row: n locks spread from `a0` to `a1` (deg azimuth at the root and on the head), each:
    root (beta_r, alpha), on-head end (beta_e, alpha + drift), then hanging `hang` metres (0: none)."""
    out = []
    n = row["n"]
    for i in range(n):
        u = 0.5 if n == 1 else i / (n - 1)
        a_root = row["a"][0] + (row["a"][1] - row["a"][0]) * u
        a_end = row["ae"][0] + (row["ae"][1] - row["ae"][0]) * u if "ae" in row else a_root
        jit = rng.normal(0.0, row.get("jitter", 0.0))
        b_r = row["b"][0] + rng.normal(0.0, row.get("b_jit", 0.0))
        b_e = np.interp(u, np.linspace(0, 1, len(row["b_end"])), row["b_end"]) if isinstance(row["b_end"], (tuple, list)) else row["b_end"]
        b_e = b_e + jit
        m = 40
        t = np.linspace(0.0, 1.0, m)
        bb = b_r + (b_e - b_r) * t
        aa = a_root + (a_end - a_root) * t ** row.get("a_curve", 1.0)
        # the root dives into the cap (no end showing where the lock starts), the tip lifts off
        off = row.get("off", 0.002) + row.get("lift", 0.0) * t ** 2 - row.get("sink", 0.004) * (1.0 - t) ** 8
        P = hair.on_head(bb, aa, off)
        hl = row.get("hang", 0.0)
        if isinstance(hl, (tuple, list)):
            hl = float(np.interp(u, np.linspace(0, 1, len(hl)), hl))
        hl *= 1.0 + rng.normal(0.0, row.get("len_jit", 0.0))
        if hl > 0.0:
            T0 = ah.unit(P[-1] - P[-3])
            if "leave" in row:
                T0 = ah.unit(T0 + np.asarray(row["leave"], float))
            pull = None
            if "pull" in row:
                pull = np.array(row["pull"], float) * np.array([np.sign(np.sin(np.radians(a_end))) or 1.0, 1.0, 1.0])
            H = hair.hang(P[-1], T0, hl, max(12, int(hl / 0.006)), row.get("clear", 0.004), body, pull=pull)
            P = np.vstack([P, H[1:]])
        P = _smooth(P, 4)
        P, L = _resample(P, max(30, int(L_of(P) / 0.004)))
        out.append((P, L))
    return out


def L_of(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def lock_mesh(hair, P, row, k, rng):
    """The ribbon along P: width and thickness profiles over u (0 root -> 1 tip), flat side away from the head/body."""
    m = len(P)
    u = np.linspace(0.0, 1.0, m)
    w = np.interp(u, *zip(*row["width"])) * (1.0 + rng.normal(0.0, 0.08))
    th = np.interp(u, *zip(*row["thick"]))
    # the normal: away from the egg's centre axis (on the head) blending to away from the spine line below
    c = hair.h.world(hair.egg.c)
    radial = P - c
    axis = P - np.column_stack([np.zeros(m), np.full(m, 0.01), P[:, 2]])
    blend = np.clip((c[2] - 0.06 - P[:, 2]) / 0.05, 0.0, 1.0)[:, None]
    N = ah.unit(ah.unit(radial) * (1.0 - blend) + ah.unit(axis) * blend)
    T = ah.unit(np.gradient(P, axis=0))
    side = ah.unit(np.cross(T, N))
    V, Q, tris = ah.sweep(P, side, np.maximum(w * 0.5, 1e-4), np.maximum(th * 0.5, 1e-4), n_u=12)
    al = np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
    hv = (np.cos(al)[None, :] * (w * 0.5)[:, None]).ravel()
    hv = np.concatenate([hv, [0.0, 0.0]]) + 0.013 * k
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    hu = np.concatenate([np.repeat(s, 12), [0.0, s[-1]]])
    return V, Q, tris, hv, hu


def body_parts(patterns):
    """The suit's objects the hair falls onto (fnmatch patterns)."""
    import fnmatch
    import bpy
    return [o for o in bpy.data.objects if o.type == "MESH" and any(fnmatch.fnmatch(o.name, p) for p in patterns)]


def build(head, spec, mats, prefix, coll="Head", body_objects=None):
    import time
    import bpy
    from mathutils.bvhtree import BVHTree
    import kit
    t0 = time.time()
    hair = Hair(head, spec)
    objs = []
    # the cap
    lo, hi = np.array([-0.10, -0.125, 1.49]), np.array([0.10, 0.11, 1.70])
    lo, hi = head.world(lo), head.world(hi)
    V, Q = ah.mesh_banded(hair.cap_field, lo, hi, spec["cap"].get("res", 0.0012))
    N = head.grad(hair.cap_field, V)
    objs.append(ah._mesh(prefix + "_Hair_Cap", V, Q, mats["hair"], coll, normals=N,
                         attrs=dict(hv=V[:, 0] * 0.6 + V[:, 1] * 0.2, hu=V[:, 2])))
    # the body to fall onto
    body = None
    if body_objects is None and "body" in spec:
        body_objects = body_parts(spec["body"])
    if body_objects:
        # the OUTER surfaces only: a solidified part's inner wall would let a rope that starts inside stay inside
        muted = [m for ob in body_objects for m in ob.modifiers if m.type == "SOLIDIFY" and m.show_viewport]
        for m in muted:
            m.show_viewport = False
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        Vs, Fs, off = [], [], 0
        for ob in body_objects:
            ev = ob.evaluated_get(dg)
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
            me.calc_loop_triangles()
            tri = np.empty(len(me.loop_triangles) * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
            keep = co[:, 2] > spec.get("body_z", 0.95)
            tri = tri.reshape(-1, 3)
            tri = tri[keep[tri].all(axis=1)]
            Vs.append(co)
            Fs.append(tri + off)
            off += len(co)
            ev.to_mesh_clear()
        for m in muted:
            m.show_viewport = True
        body = BVHTree.FromPolygons([tuple(v) for v in np.vstack(Vs)], [tuple(f) for f in np.vstack(Fs)], epsilon=0.0)
    rng = np.random.default_rng(spec.get("seed", 3))
    k = 0
    allV, allQ, allT, allhv, allhu, voff = [], [], [], [], [], 0
    for gname, row in spec["locks"]:
        for side in ((1.0, -1.0) if row.get("mirror", True) else (1.0,)):
            r = dict(row)
            if side < 0:
                r["a"] = tuple(-a for a in row["a"])
                if "ae" in row:
                    r["ae"] = tuple(-a for a in row["ae"])
                if "leave" in row:
                    r["leave"] = (-row["leave"][0], row["leave"][1], row["leave"][2])
            for P, L in lock_paths(hair, r, body, rng):
                V, Q, T, hv, hu = lock_mesh(hair, P, r, k, rng)
                allV.append(V)
                allQ.append(Q + voff)
                allT.append(T + voff)
                allhv.append(hv)
                allhu.append(hu)
                voff += len(V)
                k += 1
    if allV and body is not None:
        # the ropes keep each lock's CENTRE line clear; a wide ribbon's edges can still dip into a curved shoulder: every
        # vertex below the head is lifted to `skin_clear` off the suit's outer surface along its normal
        Vall = np.vstack(allV)
        clear = spec.get("skin_clear", 0.0012)
        moved = 0
        for i in np.where(Vall[:, 2] < head.world(head.egg.c)[2] - 0.04)[0]:
            co, nrm, _, _ = body.find_nearest(tuple(Vall[i]))
            if co is None:
                continue
            co, nrm = np.array(co), np.array(nrm)
            g = (Vall[i] - co) @ nrm
            if g < clear:
                Vall[i] += nrm * (clear - g)
                moved += 1
        print("anime hair: %d lock vertices lifted clear of the suit" % moved)
        allV = [Vall]
    if allV:
        objs.append(ah._mesh(prefix + "_Hair_Locks", np.vstack(allV), np.vstack(allQ), mats["hair"], coll, tris=np.vstack(allT),
                             attrs=dict(hv=np.concatenate(allhv), hu=np.concatenate(allhu))))
    print("anime hair: %d locks in %.1fs" % (k, time.time() - t0))
    return objs
