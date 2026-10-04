"""A figure in a skin-tight bodysuit, built from a dimensions table (Blender 4.5).

The body and the suit are ONE surface: a skin-tight suit is the body's own outline, so the
trunk and sleeve rows of the table ARE the figure's shape. The construction is the workers'
(`workwear.Figure`): one half-body loft from inside the boots to the neck, welded to its
mirror at the mid-plane, and a set-in sleeve loft meeting it at the armhole plane, the body
carrying the sleeve's surface on its own side (`Figure.envelope`). What differs from the
coverall is that nothing hangs: no folds, no ease, a few broad body forms (bust, seat) as a
displacement, and the suit's design drawn ON the surface:

  seams         raised piping lines (the `seam` attribute, wearmat.mat_gloss), from polylines
  mesh panels   inset panels of any curved outline (garment.panel), framed by a piping tube
  collar        a stand collar tube round the neck, the zip running up through it
  zip, cuffs, knee pads, ankle boots with a heel

Feature positions are given as the reference's FRONT-VIEW coordinates: ("x", x, z) a point
seen at x (metres from the centre line, x >= 0) at height z on the drawing; ("phi", deg, z)
a point by its section angle (for the sides and back the drawing does not show). Front-view
x is measured on the figure the drawing shows (`dims["feature_frame"]`, the sheet-fitted
trunk rows); when this build's rows are narrower or wider (a different hip), every x is
remapped between the same inner and outer edges at its height, so the design moves with the
body.

Conventions as in workwear: faces -Y, +X is the figure's LEFT; left limbs are modelled and
mirrored. On a leg phi 0 = outer side, 90 deg = front, 180 = inner, 270 = back; on the torso
phi 0 = the left side, 90 = centre front, -90 = centre back.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix

import garment
import implicit
import kit
import parts
import wearmat
from garment import front_phi
from kit import Loft, OnLoft, band_on, box, loft_mesh, mirror, ribbon_on, seam_attr, solid, tube, unit
from workwear import CLAMP_SNAP, Figure, build_hands, phi_where_x, zip_slider

D = math.radians
FRONT = D(90)
Z = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------- materials
def materials(p, gloss, prefix):
    """One named material per region (`<prefix>_suit`, `_mesh`, `_skin`, `_eye`, `_boot`, `_sole`,
    `_metal`); every colour from the palette `p`, the suit's gloss from `gloss`."""
    g = gloss
    return dict(
        suit=wearmat.mat_gloss(prefix + "_suit", p["SUIT"], rough=g["rough"], coat=g["coat"], coat_rough=g["coat_rough"],
                               specular=g.get("specular", 0.5), piping=g.get("piping", 0.6)),
        mesh=wearmat.mat_mesh(prefix + "_mesh", p["MESH"], p["SKIN"], cell=g.get("mesh_cell", 0.0032), show=g.get("mesh_show", 0.55),
                              surface=g.get("mesh_surface", False), coat=g.get("mesh_coat", 0.0)),
        skin=wearmat.mat_skin(prefix + "_skin", p["SKIN"], p["BROW"], p["LIP_TINT"]),
        eye=wearmat.mat_eye(prefix + "_eye", p["SCLERA"], p["IRIS"]),
        boot=wearmat.mat_gloss(prefix + "_boot", p["BOOT"], rough=g.get("boot_rough", 0.20), coat=g.get("boot_coat", 0.8),
                               coat_rough=g.get("boot_coat_rough", 0.08), piping=0.5),
        sole=wearmat.mat_rubber(prefix + "_sole", p["SOLE"], rough=0.55),
        metal=wearmat.mat_metal(prefix + "_metal", p["METAL"], rough=0.25),
        # a second net for panels that show the suit, not skin, through it (`MESH_THIGH` = what shows)
        **({"mesh_thigh": wearmat.mat_mesh(prefix + "_mesh_thigh", p["MESH"], p["MESH_THIGH"], cell=g.get("mesh_thigh_cell", 0.0030),
                                           show=1.0, surface=True, coat=g.get("mesh_coat", 0.0), width=g.get("mesh_thigh_width", 0.42))}
           if "MESH_THIGH" in p else {}),
    )


# ------------------------------------------------------------------- the frame
class Frame:
    """Front-view edges of a trunk table: at each height the inner and outer x of the left
    half (inner 0 above the crotch). `remap(x, z, other)` carries a front-view x measured on
    `other` to the same place between this table's edges."""

    def __init__(self, rows):
        r = sorted(rows, key=lambda q: q[1])
        self.z = np.array([q[1] for q in r])
        cx, a = np.array([q[2] for q in r]), np.array([q[4] for q in r])
        self.inner, self.outer = np.maximum(cx - a, 0.0), cx + a

    def edges(self, z):
        return float(np.interp(z, self.z, self.inner)), float(np.interp(z, self.z, self.outer))

    def remap(self, x, z, other):
        i0, o0 = other.edges(z)
        i1, o1 = self.edges(z)
        return i1 + (x - i0) * (o1 - i1) / max(o0 - i0, 1e-6)


class Suit:
    """The figure (workwear.Figure) plus what places the suit's design on it."""

    def __init__(self, dims):
        self.d = dims
        self.F = Figure(dims)
        sb = dims.get("sleeve_back")
        if sb:  # the sleeve with its own depth to the back at some rows (the point of the elbow)
            self.F.arm = Loft([dict(p=(x, y, z), a=a, bf=b, bb=sb.get(nm, b), n=2.0) for nm, x, y, z, a, b in dims["sleeve"]])
            self.F.az = self.F.arm.t_at_z
            self.F.t_elbow = self.F.az(dims["elbow_z"])
        self.B, self.A, self.bz, self.az = self.F.body, self.F.arm, self.F.bz, self.F.az
        self.frame = Frame(dims["trunk"])
        self.ref = Frame(dims.get("feature_frame", dims["trunk"]))
        g = dims.get("garment", {})
        if any("drop" in pn for pn in g.get("mesh_panels", ())):  # teardrop panels: their outline from two ends
            g["mesh_panels"] = tuple(dict(pn, pts=self.teardrop(**pn["drop"])) if "drop" in pn else pn for pn in g["mesh_panels"])

    def teardrop(self, a, b, w, bow=0.0, blunt=1.0, cap=0.18, n=30):
        """A clean teardrop (or a lens, `blunt` 0) on the body from its round end `a` to its point `b`
        (feature points), `w` its greatest half-width (m), its centre line bowed sideways by `bow` x its
        length (+ = to the left of a -> b on the developed surface). Laid out in the surface's own
        developed metric (round x along), returned as ("phi", deg, z) outline points."""
        (pa, ta), (pb, tb) = self.phi_t(a), self.phi_t(b)
        pb = pa + kit.wrap(pb - pa)
        r0 = float(self.B.radius([0.5 * (pa + pb)], [0.5 * (ta + tb)])[0])
        A, Bp = np.array([pa * r0, ta]), np.array([pb * r0, tb])
        d = Bp - A
        ln = float(np.linalg.norm(d))
        Tn = d / ln
        Nn = np.array([-Tn[1], Tn[0]])
        C = 0.5 * (A + Bp) + bow * ln * Nn
        s = np.linspace(0.0, 1.0, n)
        cen = ((1 - s) ** 2)[:, None] * A + (2 * s * (1 - s))[:, None] * C + (s ** 2)[:, None] * Bp
        tg = np.gradient(cen, axis=0)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        nr = np.column_stack([-tg[:, 1], tg[:, 0]])
        if blunt:
            q = np.clip(s / cap, 0.0, 1.0)
            wid = w * np.sqrt(np.clip(1.0 - (1.0 - q) ** 2, 0.0, 1.0)) * np.where(s < cap, 1.0, ((1.0 - s) / (1.0 - cap)) ** 0.85)
        else:
            wid = w * np.sin(np.pi * s) ** 0.85
        up, lo = cen + wid[:, None] * nr, cen - wid[:, None] * nr
        loop = np.vstack([up[:-1], lo[::-1][:-1]])
        if not hasattr(self, "_zt"):
            ts = np.linspace(0.0, self.B.L, 1500)
            self._zt = (ts, self.B.frames(ts)[0][:, 2])
        return tuple(("phi", math.degrees(u / r0), float(np.interp(v, *self._zt))) for u, v in loop)

    # --- points --------------------------------------------------------------
    def phi_t(self, pt, on="body"):
        """A feature point -> (phi radians, t) on the body (or arm) loft."""
        kind, v, z = pt
        if on == "arm":
            return D(v), self.az(z)
        t = self.bz(z)
        if kind == "phi":
            return D(v), t
        if kind == "bx":  # a point seen at x on the BACK view (x >= 0 her left, as on the front)
            return phi_where_x(self.B, t, self.frame.remap(v, z, self.ref), back=True) - 2.0 * math.pi, t
        return front_phi(self.B, t, self.frame.remap(v, z, self.ref)), t

    def loop(self, pts, on="body", step=0.004):
        """A CLOSED outline of feature points -> [(phi_deg, t), ...], smoothed exactly as
        garment.panel smooths a panel's outline, so a seam drawn on it frames the panel."""
        O = np.array([self.phi_t(p, on) for p in pts], float)
        lo = self.A if on == "arm" else self.B
        r0 = float(lo.radius([O[:, 0].mean()], [O[:, 1].mean()])[0])
        Q = garment.smooth_loop(np.column_stack([O[:, 0] * r0, O[:, 1]]), step)
        Q = np.vstack([Q, Q[:1]])
        return [(math.degrees(u / r0), v) for u, v in Q]

    def path(self, pts, on="body", step=0.004):
        """A polyline of feature points -> [(phi_deg, t), ...] for kit.tape_attr, densified so a
        long stretch follows the surface, not a chord across it."""
        out = []
        for p0, p1 in zip(pts[:-1], pts[1:]):
            dv = abs(p1[1] - p0[1]) * (0.0015 if p0[0] == "phi" else 1.0) if p0[0] == p1[0] else 0.02
            n = max(2, int(math.hypot(dv, p1[2] - p0[2]) / step) + 1)
            for s in np.linspace(0.0, 1.0, n, endpoint=False):
                if p0[0] == p1[0]:
                    q = (p0[0], p0[1] + s * (p1[1] - p0[1]), p0[2] + s * (p1[2] - p0[2]))
                    out.append(self.phi_t(q, on))
                else:  # mixed kinds: interpolate the converted ends
                    a, b = self.phi_t(p0, on), self.phi_t(p1, on)
                    out.append((a[0] + s * kit.wrap(b[0] - a[0]), a[1] + s * (b[1] - a[1])))
        out.append(self.phi_t(pts[-1], on))
        return [(math.degrees(p), t) for p, t in out]

    # --- the body's broad forms ---------------------------------------------
    def forms(self, P, N):
        """Bust and seat: gaussian swellings along the normal (`dims["forms"]` rows: x0, z0,
        side, sx, sz, height). Front forms fade out over the centre front, where the zip runs."""
        P, N = np.asarray(P, float), np.asarray(N, float)
        ax, z = np.abs(P[..., 0]), P[..., 2]
        d = np.zeros(ax.shape)
        for x0, z0, side, sx, sz, h in self.d.get("forms", ()):
            facing = np.clip(((-N[..., 1]) if side == "front" else N[..., 1]) - 0.15, 0.0, 0.45) / 0.45
            centre = np.clip((ax - 0.006) / 0.02, 0.0, 1.0) if side == "front" else 1.0
            d += h * np.exp(-((ax - x0) / sx) ** 2 - ((z - z0) / sz) ** 2) * facing * centre
        return d

    def lift(self, P, N):
        """Everything that displaces the body's surface (for pieces laid on it)."""
        return self.forms(P, N) + self.shoulder(P, N, "body")

    def arm_lift(self, P, N):
        """What displaces the sleeve's surface for pieces laid on it (the shoulder's union; 0 without one)."""
        return self.shoulder(P, N, "arm") if self.d["garment"].get("shoulder") else np.zeros(np.asarray(P).shape[:-1])

    # --- the shoulder: torso and arm as ONE surface ---------------------------------
    # With `garment["shoulder"]` the shoulder is the SMOOTH UNION of the body loft and the sleeve
    # loft (`union_field`), its fillet k growing from k[0] at z[0] (under the arm) to k[1] at z[1]
    # and above (over the shoulder, where the deltoid flows into the torso). Inside the box `box`
    # (the left shoulder, mirrored) that surface is ONE mesh of its own (`build_shoulder`); the body
    # and sleeve lofts give way inside the box (sunk under it, then dropped), and on the box's faces
    # the union is the plain loft, so the three meet without a step. No set-in sleeve, no armhole.
    def union_k(self, z):
        sh = self.d["garment"]["shoulder"]
        (k0, k1), (z0, z1) = sh["k"], sh["z"]
        return k0 + (k1 - k0) * np.clip((np.asarray(z) - z0) / (z1 - z0), 0.0, 1.0)

    def body_sdf(self, P):
        """Signed distance (approx., m) to the body loft, measured in each point's OWN level
        section (the trunk's sections are level): continuous in height, so a near-level surface (the
        top of the shoulder) comes out smooth, where `kit.loft_sdf`'s nearest-section distance steps.
        The radial gap is scaled by the surface's slope (first order)."""
        B = self.F.body
        if not hasattr(self, "_tz"):
            ts = np.linspace(0.0, B.L, 2000)
            self._tz = (B.frames(ts)[0][:, 2], ts)
        zs, ts = self._tz

        def radial(Pp, dz=0.0):
            # the sections are level: evaluate each distinct height once (a grid has few)
            zu, inv = np.unique(Pp[:, 2] + dz, return_inverse=True)
            c, _, e1, e2, par = (a[inv] for a in B.frames(np.interp(zu, zs, ts)))
            q = Pp - c
            x1, x2 = np.einsum("ij,ij->i", q, e1), np.einsum("ij,ij->i", q, e2)
            return np.hypot(x1, x2), Loft._polar(np.arctan2(x2, x1), par)[0]

        rho, R = radial(P)
        h = 0.0015
        dR = (radial(P, h)[1] - radial(P, -h)[1]) / (2 * h)
        return (rho - R) / np.sqrt(1.0 + dR * dR)

    def union_field(self, P, parts_=False):
        """Signed distance (approx., m) to the union of torso and arm, LEFT half (|x|)."""
        P = np.asarray(P, float)
        Pa = np.column_stack([np.abs(P[:, 0]), P[:, 1:]])
        db, da = self.body_sdf(Pa), kit.loft_sdf(self.F.arm, Pa, m=500)
        u = implicit.smin(db, da, self.union_k(Pa[:, 2]))
        return (u, db, da) if parts_ else u

    def box_depth(self, P):
        """How far (m) a point lies inside the shoulder box (negative outside), left half."""
        lo, hi = (np.asarray(v, float) for v in self.d["garment"]["shoulder"]["box"])
        P = np.asarray(P, float)
        Pa = np.concatenate([np.abs(P[..., :1]), P[..., 1:]], axis=-1)
        return np.minimum((Pa - lo).min(axis=-1), (hi - Pa).min(axis=-1))

    def shoulder(self, P, N, on="body"):
        """Displacement along N from the body (or sleeve) loft onto the shoulder's surface, for the
        pieces laid on it (seams, panels): the first crossing of the union outward along N. Without
        `garment["shoulder"]` the workers' set-in sleeve (`Figure.envelope`, body side only)."""
        sh = self.d["garment"].get("shoulder")
        if not sh:
            return self.F.envelope(P, N) if on == "body" else np.zeros(np.asarray(P).shape[:-1])
        P, N = np.asarray(P, float), np.asarray(N, float)
        shp = P.shape[:-1]
        P, N = P.reshape(-1, 3), N.reshape(-1, 3)
        out = np.zeros(len(P))
        Pa = np.column_stack([np.abs(P[:, 0]), P[:, 1:]])
        Na = np.column_stack([np.sign(P[:, 0]) * N[:, 0], N[:, 1:]])
        sel = self.box_depth(Pa) > -0.004
        if not sel.any():
            return out.reshape(shp)
        p, n = Pa[sel], Na[sel]
        other = (lambda Q: kit.loft_sdf(self.F.arm, Q, m=500)) if on == "body" else self.body_sdf
        k = self.union_k(p[:, 2])
        s_max, step = sh.get("reach", 0.04), sh.get("step", 0.002)
        ss = np.arange(0.0, s_max + 1e-9, step)
        u = lambda s: implicit.smin(s, other(p + np.reshape(s, (-1, 1)) * n), k)
        U = np.stack([u(np.full(len(p), s)) for s in ss])
        pos = U > 0.0
        found = pos.any(axis=0)
        j = np.maximum(np.argmax(pos, axis=0), 1)
        idx = np.arange(len(p))
        s0, s1, u0, u1 = ss[j - 1], ss[j], U[j - 1, idx], U[j, idx]
        for _ in range(3):  # regula falsi inside the bracket
            sm = s0 - u0 * (s1 - s0) / np.where(u1 - u0 == 0, 1e-12, u1 - u0)
            um = u(sm)
            left = um <= 0.0
            s0, u0 = np.where(left, sm, s0), np.where(left, um, u0)
            s1, u1 = np.where(left, s1, sm), np.where(left, u1, um)
        s = np.where(found, s0 - u0 * (s1 - s0) / np.where(u1 - u0 == 0, 1e-12, u1 - u0), 0.0)
        out[sel] = np.clip(np.nan_to_num(s), 0.0, s_max)
        return out.reshape(shp)

    def t_join(self, phi):
        """Where the sleeve ends and the shoulder's own mesh begins, on the sleeve loft: t at each
        section angle (rad), from `shoulder["join"]` = (z at the outer side, z at the inner side), the
        line running straight between them in |phi| (a V lowest on the outside: the drawing's seam
        at the foot of the deltoid, which a piping cord covers)."""
        z0, z1 = self.d["garment"]["shoulder"]["join"]
        a = np.abs(kit.wrap(np.asarray(phi, float))) / math.pi
        zz = z0 + (z1 - z0) * a
        if not hasattr(self, "_az"):
            ts = np.linspace(0.0, self.A.L, 1500)
            self._az = (self.A.frames(ts)[0][:, 2], ts)
        zs, ts = self._az
        return np.interp(zz, zs[::-1], ts[::-1]) if zs[0] > zs[-1] else np.interp(zz, zs, ts)

    def arm_coords(self, P, m=1500, chunk=2000):
        """(phi, t, distance) of points against the sleeve loft (its nearest section, as kit.loft_sdf)."""
        A = self.F.arm
        P = np.asarray(P, float).reshape(-1, 3)
        ts = np.linspace(0.0, A.L, m)
        C, T, e1, e2, par = A.frames(ts)
        reach = 1.6 * float(par[:, :3].max())
        ph, tt, dd = (np.empty(len(P)) for _ in range(3))
        for i in range(0, len(P), chunk):
            D = P[i:i + chunk, None, :] - C[None]
            f = np.einsum("nmk,mk->nm", D, T)
            k = np.argmin(np.abs(f) + 2.0 * np.maximum(np.linalg.norm(D, axis=2) - reach, 0.0), axis=1)
            n = np.arange(len(k))
            q = D[n, k] - f[n, k][:, None] * T[k]
            x1, x2 = np.einsum("nk,nk->n", q, e1[k]), np.einsum("nk,nk->n", q, e2[k])
            ph[i:i + chunk] = np.arctan2(x2, x1)
            tt[i:i + chunk] = ts[k] + f[n, k]
            dd[i:i + chunk] = np.hypot(x1, x2) - Loft._polar(ph[i:i + chunk], par[k])[0]
        return ph, tt, dd

    def edge_lift(self, P, N):
        """The body loft's own lift onto the union, near the shoulder box's faces only (where it meets
        the shoulder's mesh, so the two agree there); a lift beyond `edge_max` is a point buried in the
        arm, left alone (it is inside the box, under the shoulder's mesh)."""
        sh = self.d["garment"]["shoulder"]
        h = sh["res"][0 if kit.RES < 0.006 else 1]
        P = np.asarray(P, float)
        dep = self.box_depth(P)
        near = (dep > -0.006) & (dep < 5.0 * h)
        out = np.zeros(P.shape[:-1])
        if near.any():
            s = self.shoulder(P[near], np.asarray(N, float)[near], "body")
            out[near] = np.where(s > sh.get("edge_max", 0.008), 0.0, s)
        return out

    def give_way(self, P):
        """For a loft mesh near the shoulder box: (sink along -N (m), drop mask). It sinks under the
        shoulder's own mesh from `h` inside the box and is dropped deeper in."""
        sh = self.d["garment"]["shoulder"]
        h = sh["res"][0 if kit.RES < 0.006 else 1]
        dep = self.box_depth(P)
        # the shoulder's mesh is itself a hair UNDER the surface near the box's faces (build_shoulder), so its
        # edge hides under the loft; the loft sinks under it farther in: they cross once, never coincide
        return sh.get("sink", 0.0008) * np.clip((dep - 1.5 * h) / (1.5 * h), 0.0, 1.0), dep > 4.0 * h


def sleeve_to_join(S, mat, coll="Suit"):
    """The sleeve from the wrist up to the shoulder's join line (Suit.t_join), its top edge exactly
    on that line: a grid whose rows stretch per column, t = u * t_join(phi)."""
    A, res = S.A, kit.RES * 0.85
    probe_t = np.repeat(np.linspace(0.0, A.L, 9), 12)
    nc = int(round(2 * math.pi * float(A.radius(np.tile(np.linspace(0, 2 * math.pi, 12), 9), probe_t).max()) / res))
    nc = max(8, nc + (-nc) % 4)
    cols = np.linspace(0.0, 2 * math.pi, nc, endpoint=False)
    tmax = S.t_join(cols)
    u = np.linspace(0.0, 1.0, int(round(tmax.max() / res)) + 1)
    PH, TT = np.meshgrid(cols, u)
    TT = TT * tmax[None, :]
    P, _ = A.pn(PH.ravel(), TT.ravel())
    R, C = PH.shape
    idx = np.arange(R * C).reshape(R, C)
    nxt = np.roll(idx, -1, axis=1)
    faces = np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], axis=-1).reshape(-1, 4)
    return kit.new_mesh("Suit_Sleeve", P, faces, None, mat, coll)


def build_shoulder(S, mat, coll="Suit"):
    """The shoulder as ONE surface (see Suit.union_field): the smooth union of torso and arm meshed
    inside the shoulder box (implicit.surface_nets, every vertex projected onto the surface), the
    body's broad forms laid on where the torso rules, given the suit's thickness, mirrored."""
    sh = S.d["garment"]["shoulder"]
    h = sh["res"][0 if kit.RES < 0.006 else 1]
    lo, hi = (np.asarray(v, float) for v in sh["box"])
    V, Q = implicit.mesh_field(S.union_field, lo, hi, h, border=False)
    # the sampled distances leave a fine ripple that a glossy suit shows: smooth it away (the open
    # edge on the box stays where the lofts meet it)
    V = implicit.taubin(V, Q, iters=sh.get("smooth", 12), fixed=implicit.boundary_mask(Q, len(V)))
    u, db, da = S.union_field(V, parts_=True)
    e = 0.25 * h
    G = np.stack([(S.union_field(V + e * a) - S.union_field(V - e * a)) / (2 * e) for a in np.eye(3)], axis=1)
    N = unit(G)
    w_body = np.clip(0.5 + 0.5 * (da - db) / np.maximum(S.union_k(V[:, 2]), 1e-9), 0.0, 1.0)
    V = V + (S.forms(V, N) * w_body)[:, None] * N
    if "join" in sh:  # the arm below the join is the sleeve's: trim there, the cut exactly ON the join line
        ph, t, dA = S.arm_coords(V)
        tj = S.t_join(ph)
        below = (t < tj) & (dA < 0.0015)  # ON the arm and below the join (not the torso beside it)
        Q = Q[~below[Q].all(axis=1)]
        used = np.zeros(len(V), bool)
        used[Q.ravel()] = True
        mv = below & used
        V[mv] = S.A.pos(ph[mv], tj[mv])
        remap = np.cumsum(used) - 1
        V, Q = V[used], remap[Q]
    # near the box's faces the patch dips a hair under the surface: its edge (and solidify rim) hides
    # under the body loft there, which sinks under the patch farther in (Suit.give_way)
    dep = S.box_depth(V)
    used = np.zeros(len(V), bool)
    used[Q.ravel()] = True
    e = 0.25 * h
    G = np.stack([(S.union_field(V + e * a) - S.union_field(V - e * a)) / (2 * e) for a in np.eye(3)], axis=1)
    V = V - (sh.get("dip", 0.0003) * np.clip(1.0 - dep / (2.5 * h), 0.0, 1.0))[:, None] * unit(G)
    obj = kit.new_mesh("Suit_Shoulder", V, Q, None, mat, coll)
    solid(obj, S.d["garment"]["cloth"], bevel=0.0)
    obj.modifiers["Solidify"].use_even_offset = False  # an open patch's ragged edge would spike
    return mirror(obj)


def pipe_attr(g, paths):
    """`pipe` attribute: SIGNED distance (metric, on the loft's developed surface) to polylines
    [(phi_deg, t), ...], stored as 0.5 + 0.5 * d / SEAM_CAP (clipped). A signed distance
    interpolates exactly across a face, so a thin line drawn from it stays unbroken wherever it
    crosses the grid (an unsigned one, `kit.tape_attr`, dots it between vertices). Past a
    line's ends the sign is positive, so no false line runs on beyond them."""
    best = np.full(g.phi.shape, kit.SEAM_CAP)
    for path in paths:
        for (p1, t1), (p2, t2) in zip(path[:-1], path[1:]):
            p1, p2 = math.radians(p1), math.radians(p2)
            ax, ay = kit.wrap(g.phi - p1) * g.r, g.t - t1
            bx, by = kit.wrap(p2 - p1) * g.r, t2 - t1
            k = (ax * bx + ay * by) / np.maximum(bx * bx + by * by, 1e-12)
            kc = np.clip(k, 0.0, 1.0)
            dist = np.hypot(ax - kc * bx, ay - kc * by)
            sgn = np.where((k > 0.0) & (k < 1.0), np.sign(bx * ay - by * ax), 1.0)
            sgn = np.where(sgn == 0, 1.0, sgn)
            closer = dist < np.abs(best)
            best = np.where(closer, sgn * dist, best)
    return 0.5 + 0.5 * np.clip(best / kit.SEAM_CAP, -1.0, 1.0)


# --------------------------------------------------------------------- the suit
def build_suit(S, M, coll="Suit"):
    d, g = S.d, S.d["garment"]
    B, A, bz, az, F = S.B, S.A, S.bz, S.az, S.F
    suit = M["suit"]
    sm = g["seams"]
    body_paths = [S.path(p) for p in sm.get("body", ())]
    arm_paths = [S.path(p, on="arm") for p in sm.get("arm", ())]
    body_paths += [S.loop(p) for p in sm.get("body_loops", ())]
    # a panel set IN the suit (rim "seam") is framed by a seam drawn on the suit round its outline
    for pn in g["mesh_panels"]:
        if pn.get("rim", g.get("panel_rim", "tube")) == "seam":
            (arm_paths if pn.get("on") == "arm" else body_paths).append(S.loop(pn["pts"], pn.get("on", "body")))

    patch = bool(g.get("shoulder"))  # the shoulder is its own mesh: the lofts give way inside its box

    def disp(gr):
        if patch:  # sinking under the shoulder's own mesh inside its box; at the box's edge ON the union
            return S.forms(gr.P, gr.N) + S.edge_lift(gr.P, gr.N) - S.give_way(gr.P)[0]
        env = gr.env = F.envelope(gr.P, gr.N)
        return S.forms(gr.P, gr.N) + env

    armhole = g.get("armhole_seam", True)  # False: no seam round the armhole (torso and arm one surface)

    def seams(gr):
        if not armhole:
            return np.zeros(gr.P.shape[:-1])
        Pd = gr.P + getattr(gr, "env", np.zeros(gr.P.shape[:-1]))[..., None] * gr.N
        return 1.0 - np.minimum(F.armhole_seam(Pd), kit.SEAM_CAP) / kit.SEAM_CAP

    piping = g.get("seam_style", "pipe") == "tube"
    if piping:  # the seams are their own geometry (piping cords on the surface), not an attribute
        build_piping(S, body_paths, arm_paths, suit, g.get("piping", (0.0008, 0.0001)), coll)
        body_paths, arm_paths = [], []
    snap = dict(CLAMP_SNAP)
    if patch:
        snap["drop"] = lambda P: CLAMP_SNAP["drop"](P) | S.give_way(P)[1]
    suit_end = d["boot"].get("implicit", {}).get("suit_end")
    if suit_end:  # inside an implicit boot the suit's leg ends (its foot is the boot's)
        drop0 = snap["drop"]
        snap["drop"] = lambda P: drop0(P) | (P[:, 2] < suit_end)
    body = loft_mesh("Suit_Body", B, res=kit.RES * 0.85, mat=suit, coll=coll, disp=disp,
                     attrs=dict(seam=seams, pipe=lambda gr: pipe_attr(gr, body_paths)), **snap)
    mirror(solid(body, g["cloth"], bevel=0.0), merge=True)

    # the sleeve, set in at the armhole (as the coverall's), smooth
    T_root = unit(A.pos([0.0], [A.L])[0] - A.pos([0.0], [A.L - 0.05])[0])

    def onto_armhole(P):
        s = F.s_arm(P)
        k = np.where(s < 0.0, -s / float(T_root @ F.arm_n), 0.0)
        return P + k[:, None] * T_root

    def sleeve_seams(gr):
        if not armhole:
            return np.zeros(gr.P.shape[:-1])
        return 1.0 - np.minimum(F.armhole_seam(gr.P, on_sleeve=True), kit.SEAM_CAP) / kit.SEAM_CAP

    if patch:
        sleeve = sleeve_to_join(S, suit, coll)
        build_shoulder(S, suit, coll)
    else:
        sleeve = loft_mesh("Suit_Sleeve", A, res=kit.RES * 0.85, mat=suit, coll=coll,
                           attrs=dict(seam=sleeve_seams, pipe=lambda gr: pipe_attr(gr, arm_paths)),
                           drop=lambda P: F.s_arm(P) < 0.0, post=onto_armhole)
    mirror(solid(sleeve, g["cloth"], bevel=0.0))

    # cuff bands at the wrists
    c0, c1 = az(g["cuff"][0]), az(g["cuff"][1])
    mirror(band_on("Suit_Cuff", A, c0, c1, offset=g.get("cuff_lift", 0.0010), thick=g.get("cuff_thick", 0.0025), mat=suit, coll=coll,
                   bevel=0.0008,
                   attrs=dict(seam=lambda gr: seam_attr(gr, ts=[c0 + 0.0018, c1 - 0.0018]))))

    # the stand collar: a tube round the neck, rising from the neckline to under the jaw
    col = g["collar"]
    CL = Loft([parts.ring(z, 0.0, cy, a, bf, bb, n, 1) for z, cy, a, bf, bb, n in col["rings"]])
    collar = loft_mesh("Suit_Collar", CL, res=kit.RES * 0.7, mat=suit, coll=coll,
                       attrs=dict(seam=lambda gr: seam_attr(gr, ts=[CL.L - col.get("top_seam", 0.006)])))
    solid(collar, col.get("thick", 0.003), bevel=0.0012)

    if g["zip"].get("teeth"):
        build_zips(S, M, CL, coll)
    else:
        build_tape_zip(S, M, CL, coll)

    build_panels(S, M, coll)
    kp = g.get("knee_pad")
    if kp and "layers" in kp:  # layered knee pads: thin domed panels stacked over the front of the knee
        for i, ly in enumerate(kp["layers"]):
            ph, t = S.phi_t(("x", ly["x"], ly["z"]))
            mirror(garment.patch("Suit_Knee_Pad_%d" % i, OnLoft(B, ph, t), ly["hs"], ly["ht"], offset=ly["offset"], thick=ly["thick"],
                                 n=ly.get("n", 2.4), dome=ly.get("dome", 0.002), inset=ly.get("inset", 0.004), point=ly.get("point", 0.0),
                                 mat=suit, coll=coll))
    elif kp:  # shaped knee pads: a domed panel over the knee cap
        ph, t = S.phi_t(("x", kp["x"], kp["z"]))
        mirror(garment.patch("Suit_Knee_Pad", OnLoft(B, ph, t), kp["hs"], kp["ht"], offset=kp.get("offset", 0.0030), thick=0.0035,
                             n=kp.get("n", 2.4), dome=kp.get("dome", 0.003), inset=kp.get("inset", 0.004), mat=suit, coll=coll))


def build_piping(S, body_paths, arm_paths, mat, spec, coll):
    """Seams as piping cords: a thin round tube laid along each path ON the figure's displaced
    surface (the body's forms and the sleeve envelope included), sunk so only a cord of it stands
    proud. A signed-distance attribute (pipe_attr) draws false lines wherever its sign flips (past a
    line's ends, between two lines, inside a narrow loop); a cord cannot. `spec` = (radius, how far
    its centre sits above the surface)."""
    r, lift = spec
    k = 0
    for lo, paths, on in ((S.B, body_paths, "body"), (S.A, arm_paths, "arm")):
        for path in paths:
            ph = np.radians([p for p, _ in path])
            tt = np.array([t for _, t in path], float)
            # resample every 2 mm along the surface
            rr = lo.radius(ph, tt)
            u = np.concatenate([[0.0], np.cumsum(np.hypot(kit.wrap(np.diff(ph)) * rr[:-1], np.diff(tt)))])
            if u[-1] < 0.004:
                continue
            q = np.linspace(0.0, u[-1], max(4, int(u[-1] / 0.002) + 1))
            phu = np.interp(q, u, np.unwrap(ph))
            P, N = lo.pn(phu, np.interp(q, u, tt))
            if on == "body":
                P = P + np.asarray(S.lift(P, N), float)[:, None] * N
            else:
                P = P + np.asarray(S.arm_lift(P, N), float)[:, None] * N
            closed = abs(path[0][0] - path[-1][0]) < 1e-6 and abs(path[0][1] - path[-1][1]) < 1e-6
            if closed:
                P, N = P[:-1], N[:-1]
            P = P + lift * N
            if on == "body":
                P[:, 0] = np.maximum(P[:, 0], 0.0)  # the half body's cords end on the mid-plane
            mirror(tube("Suit_Piping_%03d" % k, P, normals=N, r=r, closed=closed, n_u=8, mat=mat, coll=coll))
            k += 1


def build_tape_zip(S, M, CL, coll):
    """The stage-1 zip: a flat metal strip and a hanging pull (kept for tables without `teeth`)."""
    B, bz, g = S.B, S.bz, S.d["garment"]
    col = g["collar"]
    zp = g["zip"]
    z_neck = B.pos([FRONT], [B.L])[0][2]
    zs = np.linspace(z_neck - 0.002, zp["bottom_z"], 60)
    path = [(math.degrees(front_phi(B, bz(z))), bz(z)) for z in zs]
    # (the body's forms fade out over the centre front, so the zip lies on the plain loft)
    ribbon_on("Suit_Zip_Teeth", B, path, zp["width"], offset=0.0012, thick=0.0016, mat=M["metal"], coll=coll)
    cpath = [(90.0, CL.t_at_z(z)) for z in np.linspace(col["rings"][-1][0] - 0.003, z_neck - 0.006, 12)]
    ribbon_on("Suit_Zip_Teeth_Collar", CL, cpath, zp["width"], offset=0.0012, thick=0.0016,
              mat=M["metal"], coll=coll)
    stop = OnLoft(B, front_phi(B, bz(zp["bottom_z"])), bz(zp["bottom_z"]))
    kit.box_at("Suit_Zip_Stop", stop, (0.009, 0.006, 0.003), sink=-0.0010, bevel=0.0008, seg=2, mat=M["metal"], coll=coll)
    zip_slider(CL, FRONT, CL.t_at_z(zp.get("pull_z", col["rings"][-1][0] - 0.012)), 0.004, M["metal"], coll, "Suit_Zip_Pull")



def build_panels(S, M, coll):
    """Mesh insert panels. Rim "tube" (stage 1): the panel stands proud, framed by a piping tube.
    Rim "seam": the panel is set IN the suit (a hair proud of it, so it never z-fights), its edge a
    seam drawn on the suit itself (build_suit), its net laid in the panel's own surface coordinates."""
    g = S.d["garment"]
    for i, pn in enumerate(g["mesh_panels"]):
        on = pn.get("on", "body")
        lo = S.A if on == "arm" else S.B
        outline = [S.phi_t(p, on) for p in pn["pts"]]
        name = "Suit_Mesh_%s" % pn.get("name", i)
        rim = pn.get("rim", g.get("panel_rim", "tube"))
        lift = None if on == "arm" else S.lift
        if rim == "seam":
            obj, _ = garment.panel(name, lo, outline, offset=g.get("panel_offset", 0.0003), thick=g.get("panel_thick", 0.0004), lift=lift,
                                   mat=M[pn.get("mat", "mesh")], coll=coll, surface_uv=True)
            obj.modifiers["Solidify"].use_even_offset = False  # thin sliver triangles at a tip would spike
            mirror(obj)
            continue
        obj, edge = garment.panel(name, lo, outline, offset=0.0006, thick=0.0012, lift=lift, mat=M["mesh"], coll=coll)
        mirror(obj)
        mirror(tube(name + "_Piping", edge, r=pn.get("piping", 0.0012), closed=True, n_u=8, mat=M["suit"], coll=coll))


def zip_teeth(name, loft, path, width, pitch, lift, mat, coll, tooth=(0.0016, 0.0013)):
    """A zip's teeth along a path [(phi_deg, t), ...] on a loft: small blocks a `pitch` apart,
    alternating from the two tapes (each reaching a little past the centre line), one mesh."""
    import bmesh
    ph, tt = np.radians([p for p, _ in path]), np.array([t for _, t in path], float)
    r = float(loft.radius(ph[:1], tt[:1])[0])
    u = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ph) * r, np.diff(tt)))])
    q = np.arange(0.5 * pitch, u[-1], pitch)
    P, N = loft.pn(np.interp(q, u, ph), np.interp(q, u, tt), lift)
    T = unit(np.gradient(P, axis=0))
    bm = bmesh.new()
    along, high = tooth
    for i, (p, n, tg) in enumerate(zip(P, N, T)):
        side = unit(np.cross(n, tg))
        c = p + (0.12 if i % 2 else -0.12) * width * side + 0.5 * high * n
        res = bmesh.ops.create_cube(bm, size=1.0)
        for v in res["verts"]:
            x, y, z = v.co
            v.co = (c + x * 0.62 * width * side + y * along * tg + z * high * n).tolist()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    return kit.link(bpy.data.objects.new(name, me), coll)


def zip_pull(name, loft, phi, t, lift, mat, coll, k=1.0):
    """A zip slider at the top of its track with the pull lying FLAT on the zip, pointing down:
    nothing stands off the suit. `k` scales it (a fine zip has a small slider)."""
    Fm = kit.anchor_matrix(OnLoft(loft, phi, t), lift=lift)
    return [box(name, (0.0085 * k, 0.010 * k, 0.0030 * k), M=Fm @ Matrix.Translation((0, 0.0, 0.0015 * k)), bevel=0.0010 * k, seg=2, mat=mat,
                coll=coll, taper=(0.8, 0.85)),
            box(name + "_Tab", (0.0050 * k, 0.014 * k, 0.0011 * k), M=Fm @ Matrix.Translation((0, -0.010 * k, 0.0034 * k)), bevel=0.0004 * k, seg=2,
                mat=mat, coll=coll)]


def build_zips(S, M, CL, coll):
    """The front zip (collar top -> below the navel) and the back zip (collar top -> the small of
    the back, `back_bottom_z`): a dark tape, metal teeth, a slider with a flat pull at the top."""
    B, bz, g = S.B, S.bz, S.d["garment"]
    zp, col = g["zip"], g["collar"]
    z_neck = B.pos([FRONT], [B.L])[0][2]
    z_top = col["rings"][-1][0]
    for side, phi_c, z_end in (("Front", FRONT, zp["bottom_z"]), ("Back", -FRONT, zp.get("back_bottom_z"))):
        if z_end is None:
            continue
        zs = np.linspace(z_neck - 0.001, z_end, 90)
        if side == "Front":
            body = [(math.degrees(front_phi(B, bz(z))), bz(z)) for z in zs]
        else:
            body = [(-90.0, bz(z)) for z in zs]
        neck = [(math.degrees(phi_c), CL.t_at_z(z)) for z in np.linspace(z_top - 0.002, col["rings"][0][0] + 0.004, 16)]
        for lo, path, sfx in ((B, body, ""), (CL, neck, "_Collar")):
            ribbon_on("Suit_Zip_Tape_%s%s" % (side, sfx), lo, path, zp["tape"], offset=0.0004, thick=0.0008, mat=M["suit"], coll=coll)
            zip_teeth("Suit_Zip_Teeth_%s%s" % (side, sfx), lo, path, zp["width"], zp["pitch"], 0.0002, M["metal"], coll,
                      tooth=zp.get("tooth", (0.0016, 0.0013)))
        stop = OnLoft(B, phi_c if side == "Back" else front_phi(B, bz(z_end)), bz(z_end))
        kit.box_at("Suit_Zip_Stop_%s" % side, stop, (0.007, 0.005, 0.0025), sink=-0.0008, bevel=0.0007, seg=2, mat=M["metal"], coll=coll)
        zip_pull("Suit_Zip_Pull_%s" % side, CL, phi_c, CL.t_at_z(z_top - 0.007), 0.0002, M["metal"], coll, k=zp.get("pull_scale", 1.0))


# --------------------------------------------------------------------- boots
class BootFrame:
    def __init__(self, spec):
        self.s = spec
        c, sn = math.cos(D(spec["toe_out"])), math.sin(D(spec["toe_out"]))
        self.BX, self.BY = np.array([c, sn, 0.0]), np.array([-sn, c, 0.0])
        self.ankle = np.array([spec["ankle"][0], spec["ankle"][1], 0.0])

    def w(self, x, y, z):
        return self.ankle + x * self.BX + y * self.BY + z * Z


def build_boots(S, M, coll="Boots"):
    """Ankle boots with a heel: a foot pitched on its heel (one loft, toe -> heel), a shaft
    round the ankle whose top dips in a V at the front, a sole under the forefoot and a heel
    block; the toe cap and the front panel are seams."""
    bt = BootFrame(S.d["boot"])
    s, w = bt.s, bt.w
    boot, sole_m = M["boot"], M["sole"]
    prof = s["foot"]  # (y, z centre, half-width, up, down)
    foot = Loft([dict(p=w(0, y, zc), a=a, bf=bu, bb=bd, n=(2.4, 4.0)) for y, zc, a, bu, bd in prof], front=Z)
    y0 = prof[0][0]
    ty = lambda y: float(np.interp(y, [r[0] for r in prof], foot.tk))
    mirror(loft_mesh("Boot_Foot", foot, res=0.003, cap0=True, cap1=True, mat=boot, coll=coll,
                     attrs=dict(seam=lambda g: seam_attr(g, ts=[(ty(s["toe_cap_y"]), FRONT, D(75))],
                                                         phis=[(FRONT + D(s["panel_phi"]), ty(s["toe_cap_y"]), foot.L),
                                                               (FRONT - D(s["panel_phi"]), ty(s["toe_cap_y"]), foot.L)]))))
    # sole under the forefoot, following the foot's underside, a few mm proud all round
    sy = [r for r in prof if r[0] <= s["sole_end_y"]]
    sole = Loft([dict(p=w(0, y, zc - bd + s["sole"] * 0.5 - 0.001), a=a + 0.003, bf=s["sole"] * 0.5, bb=s["sole"] * 0.5, n=6.0)
                 for y, zc, a, bu, bd in sy], front=Z)
    mirror(loft_mesh("Boot_Sole", sole, res=0.003, cap0=True, cap1=True, mat=sole_m, coll=coll))
    # heel block: from the floor to the heel's underside, narrower at the floor
    hy, hw, hd = s["heel"]
    zh = float(np.interp(hy, [r[0] for r in prof], [r[1] - r[4] for r in prof]))
    Mh = Matrix.Translation(tuple(w(0, hy, 0.5 * zh))) @ Matrix.Rotation(D(s["toe_out"]), 4, "Z")
    mirror(box("Boot_Heel", (hw * 0.80, hd * 0.80, zh + 0.004), M=Mh, taper=(1.25, 1.25), bevel=0.0025, seg=2, mat=sole_m, coll=coll))
    # shaft: a band hugging the leg (the body loft's own leg, lifted), from inside the foot up
    # round the ankle; its top dips in a V at the front (post), a shell
    B, bz = S.B, S.bz
    z0, z1 = s["shaft_z"]
    vd, vw = s["v_depth"], D(s["v_half_angle"])
    axis = w(0, 0.0, 0.0)

    def v_top(P):
        v = P[:, :2] - axis[:2]
        c = (v @ (-bt.BY[:2])) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        ang = np.arccos(np.clip(c, -1.0, 1.0))
        dip = vd * np.clip(1.0 - ang / vw, 0.0, 1.0) ** 1.3
        k = np.clip((P[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
        return np.column_stack([P[:, :2], P[:, 2] - dip * k])

    t0, t1 = bz(z0), bz(z1)
    kw = {}
    fe = s.get("feather")
    if fe:  # the shaft's top thins into the leg: its last `fe` metres sink to just under the suit, so the
        # suit meets the boot's edge flush (a seam line, not a step or a gap)
        kw["disp"] = lambda g: -(s["shaft_ease"] + 0.0004) * np.clip((g.t - (t1 - fe)) / fe, 0.0, 1.0) ** 1.5
    shaft_ob = band_on("Boot_Shaft", B, t0, t1, offset=s["shaft_ease"], thick=s.get("shaft_thick", 0.0025), bevel=0.0010, mat=boot,
                       coll=coll, post=v_top, **kw,
                       attrs=dict(seam=lambda g: seam_attr(g, ts=[t1 - 0.004], phis=[(FRONT + D(s["panel_phi"]), t0, bz(z1 - vd)),
                                                                                     (FRONT - D(s["panel_phi"]), t0, bz(z1 - vd))])))
    mirror(shaft_ob)


def build_boots_implicit(S, M, coll="Boots"):
    """A sleek ankle boot as ONE closed surface (implicit.py), when `boot["implicit"]` is given: the
    shaft is the leg's own surface eased out a little (it hugs the calf and runs into a slim ankle),
    smoothly joined to a heel cup and a slim foot (a chain of sections, heel -> toe), the whole cut
    flat underneath on the sole line; its top edge level with a shallow dip at the front. A thin sole
    under the forefoot and a block heel (`heel` = (y, half-width, half-depth, height taper)) are their
    own parts. A fine seam runs up the back (the `seam` attribute)."""
    bt = BootFrame(S.d["boot"])
    s, im = bt.s, S.d["boot"]["implicit"]
    BX, BY = bt.BX, bt.BY
    ank = bt.ankle

    def local(P):
        q = P - ank
        return np.column_stack([q @ BX, q @ BY, q[:, 2]])

    nodes = [(np.array([0.0, y, zc]), a) for y, zc, a, _ in im["foot"]]
    ratios = [hh / a for _, _, a, hh in im["foot"]]
    z_top, dip, dip_w = im["top"]
    ease0, ease1 = im["ease"]
    (yb0, zb0), (yb1, zb1), (yb2, zb2) = im["sole_line"]

    def bottom(y):  # the underside: the heel's seat, down the arch to the ball, the toe sprung a little
        return np.interp(-y, [-yb0, -yb1, -yb2], [zb0, zb1, zb2])

    def foot_d(L):
        d = None
        for (c0, r0), (c1, r1), q0, q1 in zip(nodes[:-1], nodes[1:], ratios[:-1], ratios[1:]):
            e = implicit.capsule(L, c0, c1, r0, r1, np.array([0.0, 0.0, 1.0]), ratio=(q0, q1), n=im.get("n", 2.6))
            d = e if d is None else implicit.smin(d, e, 0.003)
        return d

    def field(P):
        P = np.asarray(P, float)
        L = local(P)
        Pa = np.column_stack([np.abs(P[:, 0]), P[:, 1:]])
        z = P[:, 2]
        ease = ease0 + (ease1 - ease0) * np.clip((z_top - z) / (z_top - 0.17), 0.0, 1.0)
        shaft = np.ones(len(P))
        up = z > im["shaft_low"] - 0.03  # the leg's distance only where the shaft can be
        shaft[up] = S.body_sdf(Pa[up]) - ease[up]
        shaft = implicit.smax(shaft, im["shaft_low"] - z, 0.012)
        d = implicit.smin(shaft, foot_d(L), im["k"])
        d = implicit.smax(d, bottom(L[:, 1]) - L[:, 2], 0.0015)                 # flat underneath
        ang = np.arctan2(-L[:, 1], np.abs(L[:, 0]) + 1e-9)                      # 90 deg = straight ahead
        top = z_top - dip * np.clip(1.0 - np.abs(ang - np.pi / 2) / dip_w, 0.0, 1.0) ** 1.5
        return np.maximum(d, z - top)                                            # the top edge, cut level

    lo = ank + np.array([-0.06, -0.20, -0.005])
    hi = ank + np.array([0.06, 0.10, z_top + 0.01])
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    h = im["res"][0 if kit.RES < 0.006 else 1]
    V, Q = implicit.mesh_field(field, lo - 0.01, hi + 0.01, h)
    V = implicit.taubin(V, Q, iters=4)
    L = local(V)
    back = np.clip((L[:, 1] - 0.0) / 0.01, 0.0, 1.0)                              # the back half only
    seam = 1.0 - np.minimum(np.where(back > 0.5, np.abs(L[:, 0]), 1.0), kit.SEAM_CAP) / kit.SEAM_CAP
    mirror(kit.new_mesh("Boot", V, Q, None, M["boot"], coll, attrs=dict(seam=seam)))

    # the sole under the forefoot and the block heel (sole material), both closed
    sole_t, sole_end = im["sole"]
    hy, hw, hd, ht = im["heel"]

    def sole_field(P):
        L = local(np.asarray(P, float))
        zc_foot = np.interp(-L[:, 1], [-n[0][1] for n in nodes], [n[0][2] for n in nodes])
        zb = bottom(L[:, 1])
        foot = foot_d(np.column_stack([L[:, 0], L[:, 1], zb + 0.4 * np.maximum(zc_foot - zb, 0.0)]))
        # the footprint (the foot's outline just above its underside) a hair wider, a slab sole_t thick
        zc = bottom(L[:, 1]) - 0.5 * sole_t + 0.0008
        d = implicit.smax(foot - 0.0008, np.abs(L[:, 2] - zc) - 0.5 * sole_t, 0.0015)
        return implicit.smax(d, L[:, 1] - sole_end, 0.004)

    def heel_field(P):
        L = local(np.asarray(P, float))
        zt = bottom(L[:, 1])
        f = np.clip(L[:, 2] / max(zb0, 1e-6), 0.0, 1.0)
        k = ht + (1.0 - ht) * f                                                  # narrower at the floor
        e = ((np.abs(L[:, 0]) / (hw * k)) ** 4 + (np.abs(L[:, 1] - hy) / (hd * k)) ** 4) ** 0.25 - 1.0  # a block
        d = e * min(hw, hd)
        d = implicit.smax(d, -L[:, 2], 0.0015)
        return implicit.smax(d, L[:, 2] - zt - 0.001, 0.0015)

    for name, fld, (y0, y1), (z0, z1) in (("Boot_Sole", sole_field, (-0.20, sole_end + 0.01), (-0.004, 0.03)),
                                          ("Boot_Heel", heel_field, (hy - hd - 0.01, hy + hd + 0.01), (-0.004, zb0 + 0.01))):
        cs = [ank + x * BX + y * BY + z * Z for x in (-0.05, 0.05) for y in (y0, y1) for z in (z0, z1)]
        V, Q = implicit.mesh_field(fld, np.min(cs, axis=0), np.max(cs, axis=0), min(h, 0.0012))
        mirror(kit.new_mesh(name, V, Q, None, M["sole"], coll))


# ------------------------------------------------------------------------- head
def build_head(M, head_spec, prefix, coll="Head"):
    """The bald placeholder head: `parts.head` with this figure's spec. Its own collection and
    objects (`<prefix>_Head`, `_Eye_*`, `_Ear`), so a later stage can replace it whole."""
    loft, _ = parts.head(prefix, head_spec, dict(skin=M["skin"], eye=M["eye"]), coll=coll)
    return loft


def build_bare_hands(F, M, coll="Hands"):
    """Bare hands as one implicit surface each (`parts.bare_hand`) when the hand table carries a
    `bare` spec; otherwise the workers' lofted hands (`workwear.build_hands`)."""
    h = F.d["hand"]
    if "bare" not in h:
        return build_hands(F, M, coll)
    w = np.array(F.d["sleeve"][0][1:4], float)
    L = unit(h["down"])
    b = np.asarray(h["back"], float)
    Bv = unit(b - (b @ L) * L)
    spec = dict(h["bare"])
    if kit.RES > 0.006:  # a draft: a coarser grid
        spec["res"] = max(spec.get("res", parts.BARE_HAND["res"]), 0.0011)
    arm = unit(np.array(F.d["sleeve"][1][1:4], float) - w)  # up the forearm from the cuff's end
    return parts.bare_hand("Hand", w + L * h["drop"], L, Bv, M["skin"], coll, spec=spec, arm=arm)


def build(M, dims, head_spec, prefix):
    S = Suit(dims)
    build_suit(S, M)
    build_bare_hands(S.F, M)
    if "implicit" in S.d["boot"]:
        build_boots_implicit(S, M)
    else:
        build_boots(S, M)
    build_head(M, head_spec, prefix)
    garment.close_holes([o for o in bpy.data.objects if o.name.startswith(("Suit_Zip_Stop", "Suit_Piping"))])
    return S
