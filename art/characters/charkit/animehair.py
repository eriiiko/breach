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
        if self.sp.get("pole_guard"):
            # B2c: on the egg's vertical axis its radius is 0^(-1/n) = infinite (`Egg.rho`, the direction's azimuth undefined):
            # a grid column landing exactly there meshed a thin spike standing on the crown (B2a/B2b's "antenna")
            on = (np.abs(Q[:, 0]) < 1e-6) & (np.abs(Q[:, 1] - self.egg.c[1]) < 1e-6)
            if on.any():
                Q = Q.copy()
                Q[on, 0] = 1e-6
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
    if spec.get("kind") == "clumps":   # B2c: the layered clumps (below); B2a's blockout otherwise
        return build_clumps(head, spec, mats, prefix, coll, body_objects)
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


# ================================================================================================ B2c: layered clumps
# B2c ("the hair"): the blockout's rows replaced by LOCKS GIVEN ONE BY ONE (per-lock lists in the table), each a tapered
# clump with its own curve, laid in layers, the hanging hair placed on a traced ENVELOPE instead of dropped as ropes:
#
#   on the head  root (beta, alpha) -> optional `mid` (beta, alpha) -> `leave` (beta, alpha), or `tip` = a front-view
#                target (x, z) on the face (the fringe's tips, traced on the close-up); offset off the cap `off` = (at the
#                root, at the end) plus `bulge` x sin(pi t) (the crown's volume), the root sunk into the cap
#   falling      from the leave point down to `z` (head space) at azimuth `alpha`, its horizontal distance from the head's
#                axis r = r_skin + spread x (r_env - r_skin): `spread` 0 lies on the skin (a lock along the cheek), 1 on
#                the envelope (the outer silhouette, `envelope` = half-width W(z), front F(z), back B(z), traced on the
#                close-ups and the A-pose drawings); `spread` is keyed along the fall ((s, v), ...)
#   then         pushed off the skin, the ears, the cap and the suit (world), smoothed, re-pushed
#   section      a lens `width` x `thick` (profiles along the whole lock) whose edges `curl` towards what it lies on, so a
#                clump reads round with its edges tucked in, never a slab
#
# Every key of a lock row is either one value for all its locks or a LIST with one entry per lock (`n` of them).

def _pick(row, key, i, default=None):
    v = row.get(key, default)
    return v[i] if isinstance(v, list) else v


def _prof(p, t):
    p = np.asarray(p, float)
    return np.interp(t, p[:, 0], p[:, 1])


def _ang(Q, c):
    d = Q - c
    r = np.linalg.norm(d, axis=-1)
    beta = np.degrees(np.arcsin(np.clip(d[..., 2] / np.maximum(r, 1e-9), -1.0, 1.0)))
    alpha = np.degrees(np.arctan2(d[..., 0], -d[..., 1]))
    return beta, alpha


class Clumps:
    """The B2c hair's fields and paths (head space unless said)."""

    def __init__(self, head, spec):
        self.h, self.sp = head, spec
        self.hair = Hair(head, spec)
        self.c = head.egg.c
        ev = spec["envelope"]
        self.ez = np.asarray([r[0] for r in ev["rows"]], float)
        self.eW, self.eF, self.eB = (np.asarray([r[k] for r in ev["rows"]], float) for k in (1, 2, 3))
        self.en = ev.get("n", 2.2)
        self.ear = ah.Ear(head.F["ear"]) if "plates" in head.F.get("ear", {}) else None

    def r_env(self, alpha, z):
        o = np.argsort(self.ez)
        W = np.interp(z, self.ez[o], self.eW[o])
        F = np.interp(z, self.ez[o], self.eF[o])
        B = np.interp(z, self.ez[o], self.eB[o])
        a = np.radians(alpha)
        sa, ca = np.abs(np.sin(a)), np.cos(a)
        D = np.where(ca >= 0.0, F, B)
        n = self.en
        return ((sa / W) ** n + (np.abs(ca) / D) ** n) ** (-1.0 / n)

    def r_skin(self, alpha, z):
        """The skin's horizontal distance from the head's axis (0, cy) along azimuth alpha at height z (head space; the
        skin and neck without cuts, no ears); below `skin_floor` the value there (the neck's, inside the collar)."""
        alpha, z = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(z, float))
        a = np.radians(alpha.ravel())
        zz = np.maximum(z.ravel(), self.sp.get("skin_floor", 1.43))
        lo, hi = np.zeros(len(a)), np.full(len(a), 0.16)
        cy = self.c[1]
        for _ in range(36):
            m = 0.5 * (lo + hi)
            Q = np.column_stack([m * np.sin(a), cy - m * np.cos(a), zz])
            ins = self.h.base_h(Q) < 0.0
            lo, hi = np.where(ins, m, lo), np.where(ins, hi, m)
        return (0.5 * (lo + hi)).reshape(alpha.shape)

    def collide_w(self, P):
        """World: the head's solid for the hair to keep off (skin, ears, the cap)."""
        h = self.h
        d = np.minimum(self.hair.cap_field(P), h.base(P))
        if self.ear is not None:
            d = np.minimum(d, self.ear(h.to_head(P)) * h.s)
        return d

    def push_head(self, P, clear):
        f = self.collide_w(P)
        bad = f < clear
        if bad.any():
            g = ah.unit(self.h.grad(self.collide_w, P[bad]))
            P = P.copy()
            P[bad] -= (f[bad] - clear)[:, None] * g
        return P

    def tip_angles(self, x, z):
        """A front-view target (x, z) on the face -> (beta, alpha) about the egg's centre."""
        y = float(self.h.hit_front(self.h.base_h, [x], [z])[0])
        b, a = _ang(np.array([x, y, z]), self.c)
        return float(b), float(a)

    def path(self, row, i, rng, body):
        h = self.h
        hair = self.hair
        m = 48
        t = np.linspace(0.0, 1.0, m)
        b0, a0 = _pick(row, "root", i)
        if _pick(row, "tip", i) is not None:
            b1, a1 = self.tip_angles(*_pick(row, "tip", i))
        else:
            b1, a1 = _pick(row, "leave", i)
        mid = _pick(row, "mid", i)
        if mid is not None:   # a quadratic Bezier passing through the mid point's angles at t = 0.5
            bm, am = 2 * mid[0] - 0.5 * (b0 + b1), 2 * mid[1] - 0.5 * (a0 + a1)
            bb = (1 - t) ** 2 * b0 + 2 * t * (1 - t) * bm + t * t * b1
            aa = (1 - t) ** 2 * a0 + 2 * t * (1 - t) * am + t * t * a1
        else:
            bb = b0 + (b1 - b0) * t
            aa = a0 + (a1 - a0) * t ** _pick(row, "a_curve", i, 1.0)
        o0, o1 = _pick(row, "off", i, (0.001, 0.002))
        pw = _pick(row, "off_pow", i, 2.0)
        off = (o0 + (o1 - o0) * t ** pw + _pick(row, "bulge", i, 0.0) * np.sin(np.pi * t)
               - _pick(row, "sink", i, 0.004) * (1.0 - t) ** 8)
        Qh = h.to_head(hair.on_head(bb, aa, off))
        fall = _pick(row, "fall", i)
        if fall is not None:
            zl = Qh[-1, 2]
            zt = fall["z"] + rng.normal(0.0, fall.get("z_jit", 0.0))
            L = abs(zl - zt)
            k = max(16, int(L / 0.004))
            s = np.linspace(0.0, 1.0, k)
            al_end = fall.get("alpha", aa[-1])
            sa = s * s * (3 - 2 * s) if fall.get("ease", True) else s
            al = aa[-1] + (al_end - aa[-1]) * sa
            z = zl + (zt - zl) * s
            rs = self.r_skin(al, z)
            re = self.r_env(al, z)
            sp = _prof(fall.get("spread", ((0.0, 1.0), (1.0, 1.0))), s)
            r = rs + sp * (re - rs) + _prof(fall.get("add", ((0.0, 0.0), (1.0, 0.0))), s)
            cy = self.c[1]
            F = np.column_stack([r * np.sin(np.radians(al)), cy - r * np.cos(np.radians(al)), z])
            F += np.asarray(fall.get("shift", (0.0, 0.0, 0.0)), float)[None, :] * s[:, None] ** 1.5
            if fall.get("wave"):   # a soft S across the fall (round the head's axis), growing towards the tip
                amp, cyc = fall["wave"]
                ph = rng.uniform(0.0, 2 * np.pi)
                tang = np.column_stack([np.cos(np.radians(al)), np.sin(np.radians(al)), np.zeros(len(al))])
                F += tang * (amp * np.sin(2 * np.pi * cyc * s + ph) * s)[:, None]
            # join: the fall starts where the head part ends, the difference faded out over its first `join` (fraction)
            d0 = Qh[-1] - F[0]
            jn = fall.get("join", 0.3)
            q = np.clip(s / jn, 0.0, 1.0)
            F += d0[None, :] * (1.0 - q * q * (3 - 2 * q))[:, None]
            Qh = np.vstack([Qh, F[1:]])
        P = h.world(Qh)
        clear = _pick(row, "clear", i, 0.003)
        fix = 4
        for it in range(3):
            P[fix:] = self.push_head(P[fix:], clear)
            if body is not None and fall is not None:
                P[fix:] = hair.push(P[fix:], self.sp.get("body_clear", 0.004), body)
            P = _smooth(P, 3, fix=fix)
        P[fix:] = self.push_head(P[fix:], clear)
        P, L = _resample(P, max(30, int(L_of(P) / self.sp.get("step", 0.003))))
        return P

    def mesh(self, P, row, i, k, rng):
        """The clump along P: a lens section (`width` x `thick` over t), its edges curled by `curl` x half-width towards what
        it lies on (-N). Returns verts, quads, tris and the attributes."""
        h = self.h
        m = len(P)
        t = np.linspace(0.0, 1.0, m)
        w = _prof(_pick(row, "width", i), t) * _pick(row, "w", i, 1.0) * h.s
        th = _prof(_pick(row, "thick", i), t) * h.s
        c = h.world(self.c)
        radial = ah.unit(P - c)
        axis = ah.unit(P - np.column_stack([np.zeros(m), np.full(m, 0.01), P[:, 2]]))
        blend = np.clip((c[2] - 0.05 * h.s - P[:, 2]) / (0.05 * h.s), 0.0, 1.0)[:, None]
        N = ah.unit(radial * (1.0 - blend) + axis * blend)
        T = ah.unit(np.gradient(P, axis=0))
        S = ah.unit(np.cross(T, N))
        tw = _pick(row, "twist", i, 0.0)
        if tw:
            ang = np.radians(tw) * np.sin(np.pi * t)
            S = ah.unit(S * np.cos(ang)[:, None] + N * np.sin(ang)[:, None])
        N = ah.unit(np.cross(S, T))
        nu = 14
        al = np.linspace(0.0, 2 * math.pi, nu, endpoint=False)
        ca, sa = np.cos(al), np.sin(al)
        curl = _pick(row, "curl", i, 0.25)
        hw, hth = np.maximum(w * 0.5, 1e-4), np.maximum(th * 0.5, 1e-4)
        V = (P[:, None, :] + (hw[:, None] * ca[None, :])[:, :, None] * S[:, None, :]
             + (hth[:, None] * sa[None, :] - curl * hw[:, None] * ca[None, :] ** 2)[:, :, None] * N[:, None, :])
        V = V.reshape(-1, 3)
        idx = np.arange(m * nu).reshape(m, nu)
        nx = np.roll(idx, -1, axis=1)
        Q = np.stack([idx[:-1], nx[:-1], nx[1:], idx[1:]], axis=-1).reshape(-1, 4)
        a, b = len(V), len(V) + 1
        V = np.vstack([V, P[0] - 1e-5 * T[0], P[-1] + 1e-5 * T[-1]])
        tris = np.vstack([np.stack([np.full(nu, a), nx[0], idx[0]], axis=-1), np.stack([np.full(nu, b), idx[-1], nx[-1]], axis=-1)])
        f = Q[len(Q) // 2]
        nrm = np.cross(V[f[1]] - V[f[0]], V[f[3]] - V[f[0]])
        if nrm @ (V[f[0]] - P[len(Q) // 2 // nu]) < 0:
            Q, tris = Q[:, ::-1], tris[:, ::-1]
        across = np.concatenate([np.tile(ca, m), [0.0, 0.0]])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        hu = np.concatenate([np.repeat(s, nu), [0.0, s[-1]]])
        hv = across * np.concatenate([np.repeat(hw, nu), [0.0, 0.0]]) + 0.013 * k
        ht = np.concatenate([np.repeat(t, nu), [0.0, 1.0]])
        n = len(V)
        attrs = dict(hv=hv, hu=hu, hw=across, ht=ht,
                     ring_off=np.full(n, _pick(row, "ring", i, 0.0) + rng.normal(0.0, self.sp.get("ring_jit", 0.0))),
                     hshade=np.full(n, float(_pick(row, "shade", i, 0.0))))
        return V, Q, tris, attrs


def build_clumps(head, spec, mats, prefix, coll="Head", body_objects=None):
    """The B2c hair (see `Clumps`): the cap (as B2a's, its own table) and the locks, one object each."""
    import time
    t0 = time.time()
    C = Clumps(head, spec)
    hair = C.hair
    objs = []
    lo, hi = np.array([-0.10, -0.125, 1.49]), np.array([0.10, 0.11, 1.72])
    lo, hi = head.world(lo), head.world(hi)
    V, Q = ah.mesh_banded(hair.cap_field, lo, hi, spec["cap"].get("res", 0.0012))
    N = head.grad(hair.cap_field, V)
    nv = len(V)
    objs.append(ah._mesh(prefix + "_Hair_Cap", V, Q, mats["hair"], coll, normals=N,
                         attrs=dict(hv=V[:, 0] * 0.6 + V[:, 1] * 0.2, hu=V[:, 2], hw=np.zeros(nv), ht=np.zeros(nv),
                                    ring_off=np.zeros(nv), hshade=np.full(nv, spec["cap"].get("shade", 0.5)))))
    body = None
    if body_objects is None and "body" in spec:
        body_objects = body_parts(spec["body"])
    if body_objects:
        body = _body_bvh(body_objects, spec.get("body_z", 0.95))
    rng = np.random.default_rng(spec.get("seed", 3))
    k = 0
    allV, allQ, allT, allA, voff = [], [], [], [], 0
    for gname, row in spec["locks"]:
        n = row.get("n", 1)
        for side in ((1.0, -1.0) if row.get("mirror", True) else (1.0,)):
            r = _mirror_row(row) if side < 0 else row
            for i in range(n):
                P = C.path(r, i, rng, body)
                V, Q, T, A = C.mesh(P, r, i, k, rng)
                allV.append(V)
                allQ.append(Q + voff)
                allT.append(T + voff)
                allA.append(A)
                voff += len(V)
                k += 1
    Vall = np.vstack(allV)
    if body is not None:
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
    attrs = {key: np.concatenate([a[key] for a in allA]) for key in allA[0]}
    objs.append(ah._mesh(prefix + "_Hair_Locks", Vall, np.vstack(allQ), mats["hair"], coll, tris=np.vstack(allT), attrs=attrs))
    print("anime hair (clumps): %d locks in %.1fs" % (k, time.time() - t0))
    return objs


def _mirror_row(row):
    """Her right side: every azimuth negated (root, mid, leave, fall alpha, shift x), every front-view x negated (tip)."""
    r = dict(row)

    def neg(v, f):
        return [f(x) if x is not None else None for x in v] if isinstance(v, list) else (f(v) if v is not None else None)

    def fneg(f):
        g = dict(f)
        if "alpha" in g:
            g["alpha"] = -g["alpha"]
        if "shift" in g:
            g["shift"] = (-g["shift"][0], g["shift"][1], g["shift"][2])
        return g
    for key in ("root", "mid", "leave"):
        if key in r:
            r[key] = neg(r[key], lambda p: (p[0], -p[1]))
    if "tip" in r:
        r["tip"] = neg(r["tip"], lambda p: (-p[0], p[1]))
    if "fall" in r:
        r["fall"] = neg(r["fall"], fneg)
    if "twist" in r:
        r["twist"] = neg(r["twist"], lambda a: -a)
    return r


def _body_bvh(body_objects, body_z):
    """The suit's OUTER surfaces above `body_z` as a BVH (Solidify muted: a rope inside would stay inside its inner wall)."""
    import bpy
    from mathutils.bvhtree import BVHTree
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
        keep = co[:, 2] > body_z
        tri = tri.reshape(-1, 3)
        tri = tri[keep[tri].all(axis=1)]
        Vs.append(co)
        Fs.append(tri + off)
        off += len(co)
        ev.to_mesh_clear()
    for m in muted:
        m.show_viewport = True
    return BVHTree.FromPolygons([tuple(v) for v in np.vstack(Vs)], [tuple(f) for f in np.vstack(Fs)], epsilon=0.0)
