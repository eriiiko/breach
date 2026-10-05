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
        self.mid = MidUnion(self, dims["midplane"]) if dims.get("midplane") else None

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
    def forms(self, P, N, signed=False):
        """Bust and seat: gaussian swellings along the normal (`dims["forms"]` rows: x0, z0,
        side, sx, sz, height). Front forms fade out over the centre front, where the zip runs.
        `signed`: the LEFT half's forms continued past the mid-plane (x < 0 read as it is, not
        mirrored), so the half-body with its forms is smooth where it crosses the plane (the
        midplane union, `MidUnion`, unites it with its mirror image there)."""
        P, N = np.asarray(P, float), np.asarray(N, float)
        ax, z = (P[..., 0] if signed else np.abs(P[..., 0])), P[..., 2]
        d = np.zeros(ax.shape)
        for x0, z0, side, sx, sz, h in self.d.get("forms", ()):
            facing = np.clip(((-N[..., 1]) if side == "front" else N[..., 1]) - 0.15, 0.0, 0.45) / 0.45
            centre = np.clip((ax - 0.006) / 0.02, 0.0, 1.0) if side == "front" else 1.0
            d += h * np.exp(-((ax - x0) / sx) ** 2 - ((z - z0) / sz) ** 2) * facing * centre
        if self.d.get("midline"):
            d += self.midline_fillet(P, N)
        return d

    # --- the centre line over the pelvis -----------------------------------------
    # Above the crotch the half-body's section reaches past the centre line and is cut there (the
    # mirror weld), so front and back the two halves meet at an angle: a V crease down the centre,
    # whose weld follows the grid in steps. `dims["midline"]` = dict(front=w, back=w, z=(z_lo, z_hi),
    # fade=m, s_max=s) fills that V with a smooth fillet reaching w (m) either side, as a smooth maximum of the
    # two halves' surfaces: at the centre the surface then crosses the mid-plane square, so the two
    # halves meet with no crease. The fillet grows in over `fade` above z_lo (the crotch, where the
    # legs part) and is off where the halves meet steeper than `s_max` (the crotch itself).
    def _midline_slopes(self):
        """(z, |dy/dx| of the section where it crosses x = 0 in front, behind), nan where it does not."""
        if not hasattr(self, "_mls"):
            ml = self.d["midline"]
            zs = np.arange(ml["z"][0], ml["z"][1] + 1e-9, 0.001)
            zt, tt = self._zt_table()
            ph = np.linspace(-math.pi, math.pi, 1441)
            sf, sb = np.full(len(zs), np.nan), np.full(len(zs), np.nan)
            for i, z in enumerate(zs):
                Q = self.B.pos(ph, np.full(ph.shape, float(np.interp(z, zt, tt))))
                for half, out in ((Q[:, 1] < self.centre_y(z)[0], sf), (Q[:, 1] >= self.centre_y(z)[0], sb)):
                    q = Q[half]
                    q = q[np.argsort(q[:, 0])]
                    if q[0, 0] < 0.0 < q[-1, 0]:
                        j = int(np.searchsorted(q[:, 0], 0.0))
                        dx = q[j, 0] - q[j - 1, 0]
                        out[i] = abs((q[j, 1] - q[j - 1, 1]) / dx) if dx > 1e-9 else np.inf
            self._mls = (zs, sf, sb)
        return self._mls

    def midline_fillet(self, P, N):
        ml = self.d["midline"]
        zs, sf, sb = self._midline_slopes()
        P = np.asarray(P, float)
        ax, z = np.abs(P[..., 0]), P[..., 2]
        back = P[..., 1] >= self.centre_y(np.clip(z, zs[0], zs[-1]).ravel()).reshape(z.shape)
        s = np.where(back, np.interp(z, zs, np.nan_to_num(sb, nan=-1.0)), np.interp(z, zs, np.nan_to_num(sf, nan=-1.0)))
        w = np.where(back, ml.get("back", 0.0), ml.get("front", 0.0))
        k = 2.0 * s * w * np.clip((z - ml["z"][0]) / ml.get("fade", 0.02), 0.0, 1.0) * ((z >= zs[0]) & (z <= zs[-1]))
        on = (s > 0.0) & (s < ml.get("s_max", 3.0)) & (k > 1e-9)
        s, k = np.where(on, s, 1.0), np.where(on, k, 1.0)
        a = s * ax
        f = kit.smax(a, -a, k) - a
        return np.where(on, np.maximum(f, 0.0), 0.0)

    def lift(self, P, N):
        """Everything that displaces the body's surface (for pieces laid on it)."""
        return self.body_lift(P, N) + self.shoulder(P, N, "body")

    def body_lift(self, P, N):
        """The body loft's own displacement along N: its forms, or with `dims["midplane"]` the lift onto
        the midplane union (which carries the forms)."""
        if self.mid is not None:
            return self.mid.lift(P, N)[0]
        return self.forms(P, N)

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
        zs, ts = self._zt_table()

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
        db, da = self.body_sdf(Pa), np.ones(len(Pa))
        near = Pa[:, 0] > self.d["garment"]["shoulder"].get("arm_x", 0.0)  # nearer the neck the arm cannot reach the union
        da[near] = kit.loft_sdf(self.F.arm, Pa[near], m=500)
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
        near = (dep > -0.006) & ((dep < 5.0 * h) | ("cut" in sh))
        out = np.zeros(P.shape[:-1])
        if near.any():
            s = self.shoulder(P[near], np.asarray(N, float)[near], "body")
            out[near] = np.where(s > sh.get("edge_max", 0.008), 0.0, s)
        return out

    # --- the raglan cut: the shoulder's own mesh bounded by seam lines, not by its box -----
    def _cut_curves(self):
        """World x of the cut at each height, in front and behind (from `shoulder["cut"]` feature
        points, on the loft), and the lowest height of the cut under the arm."""
        if not hasattr(self, "_cuts"):
            c = self.d["garment"]["shoulder"]["cut"]
            cur = []
            for pts in (c["front"], c["back"]):
                path = self.path(pts, step=0.002)
                P = self.B.pos(np.radians([q for q, _ in path]), np.array([t for _, t in path]))
                o = np.argsort(P[:, 2])
                cur.append((P[o, 2], P[o, 0]))
            self._cuts = (cur[0], cur[1], c["z_low"])
        return self._cuts

    def _zt_table(self):
        if not hasattr(self, "_tz"):
            ts = np.linspace(0.0, self.B.L, 2000)
            self._tz = (self.B.frames(ts)[0][:, 2], ts)
        return self._tz

    def centre_y(self, z):
        zs, ts = self._zt_table()
        return self.B.frames(np.interp(np.asarray(z, float), zs, ts))[0][:, 1]

    def cut_x(self, y, z):
        (zf, xf), (zb, xb), _ = self._cut_curves()
        return np.where(y < self.centre_y(z), np.interp(z, zf, xf), np.interp(z, zb, xb))

    def in_torso_region(self, P):
        """Torso points (left half) the shoulder's mesh owns under a raglan cut."""
        P = np.asarray(P, float)
        x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
        z_low = self._cut_curves()[2]
        return (z >= z_low) & (x > self.cut_x(y, z)) & (self.box_depth(P) > 0.0)

    def body_onto_cut(self, P, disp):
        """Body loft vertices the shoulder owns, moved onto the cut ALONG the loft (to the cut's x at
        their height, or to its lowest line, whichever is nearer), then displaced afresh by
        `disp(P, N)`: the body's edge lies exactly on the seam line."""
        P = np.array(P, float)
        ins = self.in_torso_region(P)
        if not ins.any():
            return P
        B = self.B
        zs, ts = self._zt_table()
        p = P[ins]
        t = np.interp(p[:, 2], zs, ts)
        c, _, e1, e2, _ = B.frames(t)
        q = p - c
        ph = np.arctan2(np.einsum("ij,ij->i", q, e2), np.einsum("ij,ij->i", q, e1))
        z_low = self._cut_curves()[2]
        xc = self.cut_x(p[:, 1], p[:, 2])
        down = (p[:, 2] - z_low) < (np.abs(p[:, 0]) - xc)
        t2 = np.where(down, float(np.interp(z_low, zs, ts)), t)
        ph2 = ph.copy()
        for i in np.flatnonzero(~down):
            back = ph[i] < 0.0
            ph2[i] = phi_where_x(B, t[i], xc[i], back=back) - (2 * math.pi if back else 0.0)
        Pn, Nn = B.pn(ph2, t2)
        P[ins] = Pn + np.asarray(disp(Pn, Nn), float)[:, None] * Nn
        return P

    def give_way(self, P):
        """For a loft mesh near the shoulder box: (sink along -N (m), drop mask). It sinks under the
        shoulder's own mesh from `h` inside the box and is dropped deeper in."""
        sh = self.d["garment"]["shoulder"]
        h = sh["res"][0 if kit.RES < 0.006 else 1]
        dep = self.box_depth(P)
        # the shoulder's mesh is itself a hair UNDER the surface near the box's faces (build_shoulder), so its
        # edge hides under the loft; the loft sinks under it farther in: they cross once, never coincide
        return sh.get("sink", 0.0008) * np.clip((dep - 1.5 * h) / (1.5 * h), 0.0, 1.0), dep > 4.0 * h


# ------------------------------------------------------- the pelvis: one surface across the mid-plane
def _smoothstep(q):
    q = np.clip(q, 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q)


class MidUnion:
    """The pelvis as ONE surface across the mid-plane: the SMOOTH UNION of the left half-body and its
    own mirror image (`dims["midplane"]`), from under the crotch to the waist.

        U(P) = smin(F(P), F(mirror P), k)        F = Suit.body_sdf - Suit.forms(signed)   (implicit.smin)

    F is the left half-body with its broad forms, the forms continued past the plane with signed x, so
    it is smooth where it crosses the plane. For this smin two equal arguments give F - k/4 and
    arguments more than k apart are untouched, so ON the mid-plane the surface lies where F = k/4:
      - between the legs (half-gap g) the two thighs are bridged where g <= k/4 and untouched where
        g >= k/2; between, the gap's half-width is sqrt(k (g - k/4)): it closes in a ROUND arch of
        tip radius k |dg/dz| / 2, and under it the surface runs from front to back as one saddle;
      - above the crotch the half-section is cut at the plane at an angle (a V); the union fills it
        with a fillet reaching w either side: k = 2 w sigma, sigma = |n_x| of the half-body's normal
        where it crosses the plane (a square crossing, the waist, gets k = 0 and is left as it is;
        below the crotch's tip sigma = 1). `reach` = ((z, w front, w back), ...) is the table of w,
        zero at its ends; front and back blend over `band` about the section's centre y.
    ONE grid (the loft's, so its (phi, t) still carry the seams, piping and panels): every vertex is
    lifted along its normal onto U = 0 (`lift`); a vertex the union swallows (its ray reaches the
    plane still inside) is in the row's RUN, and `grid_post` welds the run onto the union's own
    crossing of the plane in that row (its front part onto the front crossing, the back part onto the
    back one); faces lying wholly in the run are dropped (`grid_drop`). `refine` = ((z_lo, z_hi, pitch,
    grade), ...): rows `pitch` apart between z_lo and z_hi (graded over `grade`), the finest band
    winning. The underside of the crotch is nearly LEVEL, so a level row just above the tip already
    bridges a long stretch front to back; rows a fraction of a millimetre apart there let the bridge
    grow a few columns per row (one long fan of slivers from each weld point otherwise)."""

    X_EPS = 3e-4  # a vertex this close to the plane is welded, a free one held this far off it (the mirror merge's 2e-4)

    def __init__(self, S, spec):
        self.S, self.sp = S, spec
        self.z0, self.z1 = spec["z"]
        r = np.array(spec["reach"], float)
        self.rz, self.rf, self.rb = r[:, 0], r[:, 1], r[:, 2]
        self.band = spec.get("band", 0.03)
        zs = np.arange(self.z0 - 0.01, self.z1 + 0.0105, 0.001)
        self._cy = (zs, S.centre_y(zs))
        self._sigma()

    # --- the field -------------------------------------------------------------
    def f(self, P, N):
        return self.S.forms(P, N, signed=True)

    def F(self, Q, N):
        """The left half-body with its forms (approx. signed distance, m)."""
        return self.S.body_sdf(Q) - self.f(Q, N)

    def normal(self, Q, e=4e-4):
        """The bare half-body's outward normal near Q (for the forms' facing)."""
        g = np.stack([(self.S.body_sdf(Q + e * a) - self.S.body_sdf(Q - e * a)) / (2 * e) for a in np.eye(3)], axis=1)
        return unit(g)

    def grad_F(self, Q, e=4e-4):
        N = self.normal(Q)
        return np.stack([(self.F(Q + e * a, N) - self.F(Q - e * a, N)) / (2 * e) for a in np.eye(3)], axis=1)

    def cy(self, z):
        return np.interp(z, *self._cy)

    def k(self, Q):
        z, y = Q[:, 2], Q[:, 1]
        wb = _smoothstep((y - self.cy(z)) / self.band + 0.5)
        sf, sb = np.interp(z, self._sz, self._sf), np.interp(z, self._sz, self._sb)
        kf = 2.0 * np.interp(z, self.rz, self.rf) * sf
        kb = 2.0 * np.interp(z, self.rz, self.rb) * sb
        inside = (z >= self.rz[0]) & (z <= self.rz[-1])
        return np.where(inside, (1.0 - wb) * kf + wb * kb, 0.0)

    def crossing(self, z, level=None, span=0.16, dy=0.0005):
        """Where the surface `F = level` (default the union's, k/4) crosses the mid-plane at each height z:
        (y front, y back, found) -- the front-most and back-most y on x = 0 with F <= level."""
        z = np.atleast_1d(np.asarray(z, float))
        ys = np.arange(-span, span + 1e-9, dy)
        Y = self.cy(z)[:, None] + ys[None, :]
        Q = np.stack([np.zeros(Y.shape), Y, np.repeat(z[:, None], len(ys), axis=1)], axis=-1).reshape(-1, 3)
        N = self.normal(Q)
        Fv = self.F(Q, N).reshape(Y.shape)
        lv = (self.k(Q) / 4.0).reshape(Y.shape) if level is None else np.full(Y.shape, float(level))
        g = Fv - lv
        ins = g <= 0.0
        found = ins.any(axis=1)
        yf, yb = np.full(len(z), np.nan), np.full(len(z), np.nan)
        for i in np.flatnonzero(found):
            j = np.flatnonzero(ins[i])
            j0, j1 = j[0], j[-1]
            # the crossings, interpolated between the sample outside and the one inside
            yf[i] = Y[i, j0] if j0 == 0 else Y[i, j0 - 1] + (Y[i, j0] - Y[i, j0 - 1]) * g[i, j0 - 1] / (g[i, j0 - 1] - g[i, j0])
            yb[i] = Y[i, j1] if j1 == len(ys) - 1 else Y[i, j1] + (Y[i, j1 + 1] - Y[i, j1]) * g[i, j1] / (g[i, j1] - g[i, j1 + 1])
        return yf, yb, found

    def _sigma(self):
        """sigma(z), front and back: |n_x| of the half-body (with its forms) where it crosses the plane;
        1 below the crotch's tip, where it does not."""
        zs = np.arange(self.z0 - 0.01, self.z1 + 0.0105, 0.002)
        yf, yb, found = self.crossing(zs, level=0.0)
        sf, sb = np.ones(len(zs)), np.ones(len(zs))
        for y, out in ((yf, sf), (yb, sb)):
            m = found & np.isfinite(y)
            if m.any():
                Q = np.column_stack([np.zeros(m.sum()), y[m], zs[m]])
                g = self.grad_F(Q)
                # the V is the level section's: its normal's horizontal part (so sigma -> 1 at the tip,
                # where the section's inner point touches the plane, continuous with the slot below)
                out[m] = np.abs(g[:, 0]) / np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-12)
        self._sz, self._sf, self._sb = zs, sf, sb

    # --- vertices onto the union --------------------------------------------------
    def march(self, P0, D, Nf, rate, t_max=0.024, dt=0.0008):
        """Points P0 on the half-body WITH its forms, pushed along D onto U = 0: (t, swallowed). The own
        term grows as rate * t (rate = D . N), the mirror's is F at the mirrored point (Nf the normals for
        the forms' facing). A point the union swallows (its ray reaches the plane still inside, or it starts
        at or past the plane) gets t to the plane (0 if it starts past it)."""
        mir = np.array([-1.0, 1.0, 1.0])

        def U(t, idx=slice(None)):
            Q = P0[idx] + t[:, None] * D[idx]
            return implicit.smin(rate[idx] * t, self.F(Q * mir, Nf[idx]), self.k(Q)), Q[:, 0]

        n = len(P0)
        t0 = np.zeros(n)
        u0, x0 = U(t0)
        sw = x0 <= self.X_EPS                 # at or past the plane: welded, whatever the mirror (a centred
        done = (u0 >= 0.0) & ~sw              # section's far half lies ON the mirror's surface); untouched
        out = np.zeros(n)
        hit = np.zeros(n, bool)
        prev_t, prev_u = t0.copy(), u0.copy()
        for j in range(1, int(t_max / dt) + 1):
            act = ~done & ~hit & ~sw
            if not act.any():
                break
            tj = np.full(n, j * dt)
            uj, xj = U(tj)
            root = act & (uj > 0.0)
            sw |= act & ~root & (xj <= self.X_EPS)   # reached the plane inside the union
            if root.any():
                r = np.flatnonzero(root)
                a, b, ua, ub = prev_t[r], tj[r], prev_u[r], uj[r]
                for _ in range(5):  # regula falsi inside the bracket
                    m = a - ua * (b - a) / np.where(ub - ua == 0, 1e-12, ub - ua)
                    um = U(m, r)[0]
                    lo = um <= 0.0
                    a, ua = np.where(lo, m, a), np.where(lo, um, ua)
                    b, ub = np.where(lo, b, m), np.where(lo, ub, um)
                tr = a - ua * (b - a) / np.where(ub - ua == 0, 1e-12, ub - ua)
                past = P0[r, 0] + tr * D[r, 0] <= self.X_EPS   # the root past the plane is the mirror's surface
                out[r] = tr
                hit[r[~past]] = True
                sw[r[past]] = True
            prev_t, prev_u = np.where(act, tj, prev_t), np.where(act, uj, prev_u)
        sw |= ~done & ~hit                    # no root within reach: inside
        dx = np.where(D[:, 0] < -1e-6, D[:, 0], -1e-6)
        out = np.where(sw, np.where(P0[:, 0] > 0.0, np.clip(-P0[:, 0] / dx, 0.0, t_max), 0.0), out)
        return out, sw

    def _sel(self, P):
        return (P[:, 2] >= self.rz[0]) & (P[:, 2] <= self.rz[-1]) & (P[:, 0] < 0.09)

    def lift(self, P, N):
        """(lift along N onto U = 0, swallowed mask) for loft points P with normals N: for the pieces laid
        on the surface (seams, panels, zips: `Suit.lift`). Outside the region exactly the forms."""
        P, N = np.asarray(P, float), np.asarray(N, float)
        shp = P.shape[:-1]
        P, N = P.reshape(-1, 3), N.reshape(-1, 3)
        sel = self._sel(P)
        s = self.S.forms(P, N)
        run = np.zeros(len(P), bool)
        if sel.any():
            p, n = P[sel], N[sel]
            f = self.f(p, n)
            t, sw = self.march(p + f[:, None] * n, n, n, np.ones(len(p)))
            s[sel], run[sel] = f + t, sw
        return s.reshape(shp), run.reshape(shp)

    # --- the grid (kit.loft_mesh) ------------------------------------------------------
    def rows(self, res):
        """The body loft's rows (t): `res` apart, `refine[2]` apart between refine[0] and refine[1] (graded)."""
        B = self.S.B
        self._res = res
        zt, tt = self.S._zt_table()
        ts = np.linspace(0.0, B.L, 20001)
        if not self.sp.get("refine"):
            return np.linspace(0.0, B.L, max(2, int(round(B.L / res))) + 1)
        z = np.interp(ts, tt, zt)
        dens = np.full(len(ts), 1.0 / res)
        for za, zb, pitch, grade in self.sp["refine"]:
            w = _smoothstep((z - (za - grade)) / grade) * _smoothstep(((zb + grade) - z) / grade)
            dens = np.maximum(dens, 1.0 / res + w * (1.0 / pitch - 1.0 / res))
        Phi = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(ts))])
        n = max(2, int(round(Phi[-1])))
        return np.interp(np.linspace(0.0, Phi[-1], n + 1), Phi, ts)

    def grid_disp(self, gr):
        """The body grid's displacement along N (rows, cols): its forms. The union then pushes each vertex
        HORIZONTALLY, along its section's own normal, onto U = 0, so every row stays a level section of the
        union (pushed along the 3D normal, the vertices near the crotch, whose normals point down, would land
        below the next row's and fold the mesh); push and run stashed for grid_drop / grid_post."""
        P, N = gr.P.reshape(-1, 3), gr.N.reshape(-1, 3)
        R, C = gr.P.shape[:2]
        f = self.S.forms(P, N)
        sel = self._sel(P)
        push = np.zeros((len(P), 3))
        run = np.zeros(len(P), bool)
        if sel.any():
            p, n = P[sel], N[sel]
            f[sel] = self.f(p, n)
            nh = np.column_stack([n[:, 0], n[:, 1], np.zeros(len(n))])
            ln = np.maximum(np.linalg.norm(nh, axis=1), 1e-6)
            nh /= ln[:, None]
            t, sw = self.march(p + f[sel][:, None] * n, nh, n, ln)
            push[sel], run[sel] = t[:, None] * nh, sw
        run = run.reshape(R, C)
        zr = gr.P[:, :, 2].mean(axis=1)
        region = (zr >= self.rz[0]) & (zr <= self.rz[-1])
        run &= region[:, None]
        A = np.full((R, 3), np.nan)
        Bk = np.full((R, 3), np.nan)
        rr = np.flatnonzero(run.any(axis=1))
        if len(rr):
            yf, yb, found = self.crossing(zr[rr])
            for i, r in enumerate(rr):
                if found[i]:
                    A[r], Bk[r] = (0.0, yf[i], zr[r]), (0.0, yb[i], zr[r])
                else:  # no bridge in this row (the slot is open, if only just): nothing is welded or dropped
                    run[r] = False
            close = np.linalg.norm(A - Bk, axis=1) < 3e-4  # a bridge one point wide: one weld point
            mid = 0.5 * (A + Bk)
            A[close], Bk[close] = mid[close], mid[close]
        Q = (P + f[:, None] * N + push).reshape(R, C, 3)
        fix = np.full((R, C, 3), np.nan)
        for r in np.flatnonzero(run.any(axis=1)):
            self._spread(Q[r], run[r], A[r], Bk[r], fix[r], self.sp.get("spread", 1.0) * getattr(self, "_res", 0.0034))
        self._grid = dict(run=run, region=region, A=A, B=Bk, push=push.reshape(R, C, 3), fix=fix)
        return f.reshape(R, C)

    def relax(self, obj):
        """The tip of the arch, where the union's surface is nearly LEVEL: level rows meet it at a grazing
        angle, and the smallest error in a vertex's push folds the rows there. `relax` = (z_lo, z_hi, x_max,
        rounds): the half-body mesh's vertices in that box (left half, x < x_max) are smoothed (Taubin) and
        projected back onto U = 0 by Newton steps along its gradient, `rounds` times; the weld vertices stay
        in the plane (smoothed and projected within it), the box's border vertices stay where they are."""
        spec = self.sp.get("relax")
        if not spec:
            return
        z_lo, z_hi, x_max, rounds = spec
        me = obj.data
        V = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", V)
        V = V.reshape(-1, 3)
        E = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", E)
        E = E.reshape(-1, 2)
        box = (V[:, 0] < x_max) & (V[:, 2] > z_lo) & (V[:, 2] < z_hi)
        # the box's border: vertices in it with a neighbour outside
        out_nb = np.zeros(len(V), bool)
        for a, b in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
            out_nb[a[~box[b]]] = True
        move = box & ~out_nb
        on = np.abs(V[:, 0]) < 1e-7
        if not move.any():
            return
        sub = np.flatnonzero(box)
        keep = np.isin(E[:, 0], sub) & np.isin(E[:, 1], sub)
        Es = E[keep]
        deg = np.bincount(Es.ravel(), minlength=len(V)).astype(float)
        e = 2e-4

        def project(idx, iters=4):
            for _ in range(iters):
                Q = V[idx]
                u = self.U(Q)
                g = np.stack([(self.U(Q + e * a) - self.U(Q - e * a)) / (2 * e) for a in np.eye(3)], axis=1)
                g[on[idx], 0] = 0.0  # a weld vertex moves within the plane
                step = (u / np.maximum(np.sum(g * g, axis=1), 1e-12))[:, None] * g
                n = np.linalg.norm(step, axis=1, keepdims=True)
                V[idx] -= step * np.minimum(1.0, 5e-4 / np.maximum(n, 1e-12))

        idx = np.flatnonzero(move)
        for _ in range(rounds):
            for f in (0.5, -0.53, 0.5, -0.53):
                S_ = np.zeros_like(V)
                np.add.at(S_, Es[:, 0], V[Es[:, 1]])
                np.add.at(S_, Es[:, 1], V[Es[:, 0]])
                L = S_ / np.maximum(deg, 1.0)[:, None] - V
                L[on, 0] = 0.0
                V[idx] += f * L[idx]
            project(idx)
        x = V[idx, 0]
        V[idx, 0] = np.where(on[idx], 0.0, np.where(x < 2 * self.X_EPS, self.X_EPS + np.maximum(x, 0.0) ** 2 / (4 * self.X_EPS), x))
        me.vertices.foreach_set("co", V.ravel())
        me.update()

    def U(self, Q):
        """The union field at points Q (left half)."""
        N = self.normal(Q)
        return implicit.smin(self.F(Q, N), self.F(Q * np.array([-1.0, 1.0, 1.0]), N), self.k(Q))

    def _spread(self, Q, run, A, B, fix, h):
        """One row: next to each weld point the union's section hugs the plane (it leaves it as a square
        root) for a stretch that no free vertex lands on -- their rays went into the bridge -- so the face
        from the weld point to the first free vertex would lie almost IN the plane (a fin, dark in a
        render). The run's columns nearest each end are taken out of the run and spread along the section
        between the weld point and the first free vertex, `h` apart (each on U = 0, found square to the
        chord), one column at least staying on the weld point. Writes their positions into `fix`."""
        C = len(run)
        starts = np.flatnonzero(~np.roll(run, 1) & run)
        if len(starts) != 1 or run.all():
            return
        c0 = starts[0]
        arc = (c0 + np.arange(int(run.sum()))) % C
        ends = ((arc, (c0 - 1) % C), (arc[::-1], (arc[-1] + 1) % C))
        Pn0, Pn1 = Q[ends[0][1]], Q[ends[1][1]]
        # which end of the run meets which weld point: the nearer pairing
        if np.linalg.norm(Pn0 - A) + np.linalg.norm(Pn1 - B) > np.linalg.norm(Pn0 - B) + np.linalg.norm(Pn1 - A):
            A, B = B, A
        used = 0
        for (cols, nb), W in zip(ends, (A, B)):
            Pn = Q[nb]
            L = float(np.linalg.norm(Pn[:2] - W[:2]))
            m = int(min(max(0, math.ceil(L / h) - 1), max(0, (len(arc) - used) // 2 - 1)))
            if m == 0:
                continue
            d = (Pn - W) / max(L, 1e-9)
            nrm = np.array([-d[1], d[0], 0.0])
            mid = 0.5 * (Pn + W)
            if self.U(np.array([mid + 5e-4 * nrm]))[0] < self.U(np.array([mid - 5e-4 * nrm]))[0]:
                nrm = -nrm  # the side where the union is not
            u = np.arange(m, 0, -1) / (m + 1.0)      # the column next to the free vertex first
            Q0 = W + u[:, None] * (Pn - W)
            ss = np.linspace(-0.003, 0.006, 19)
            Uv = np.stack([self.U(Q0 + s * nrm) for s in ss])
            for i in range(m):
                k = np.flatnonzero((Uv[:-1, i] <= 0.0) & (Uv[1:, i] > 0.0))
                if not len(k):
                    continue
                k = k[0]
                a, b, ua, ub = ss[k], ss[k + 1], Uv[k, i], Uv[k + 1, i]
                for _ in range(4):  # regula falsi
                    m = a - ua * (b - a) / (ub - ua)
                    um = self.U(np.array([Q0[i] + m * nrm]))[0]
                    a, ua, b, ub = (m, um, b, ub) if um <= 0.0 else (a, ua, m, um)
                s = a - ua * (b - a) / (ub - ua)
                fix[cols[i]] = Q0[i] + s * nrm
                run[cols[i]] = False
            used += m

    def grid_drop(self, base):
        """drop for loft_mesh: inside the region the run, elsewhere `base`."""
        def drop(P):
            g = self._grid
            reg = np.repeat(g["region"], g["run"].shape[1])
            return np.where(reg, g["run"].ravel(), base(P))
        return drop

    def grid_post(self, base):
        """post for loft_mesh (grid): `base` (a grid post) outside the region; inside, the run welded onto
        the union's crossing of the plane and every other vertex left where its lift put it."""
        def g_(G):
            G = np.asarray(G, float)
            X = np.asarray(base(G.copy()), float).reshape(G.shape).copy()
            gd = self._grid
            run, A, B = gd["run"], gd["A"], gd["B"]
            for r in np.flatnonzero(gd["region"]):
                X[r] = G[r] + gd["push"][r]
                fx = np.isfinite(gd["fix"][r, :, 0])
                X[r, fx] = gd["fix"][r, fx]
                # a free vertex never nearer the plane than X_EPS (the mirror merge would pinch it): a C1 soft clamp
                x, e = X[r, :, 0], self.X_EPS
                X[r, :, 0] = np.where(x < 2.0 * e, e + np.maximum(x, 0.0) ** 2 / (4.0 * e), x)
                cols = np.flatnonzero(run[r])
                if not len(cols):
                    continue
                if not np.isfinite(A[r, 0]):  # a run but no bridge in this row: hold it at the plane's edge
                    X[r, cols, 0] = self.X_EPS
                    continue
                front = G[r, cols, 1] < 0.5 * (A[r, 1] + B[r, 1])
                X[r, cols[front]] = A[r]
                X[r, cols[~front]] = B[r]
            return X
        g_.grid = True
        return g_


def _clean(obj, dist=2e-5):
    """Weld the coincident vertices a trim leaves (several snapped onto the same point of a cut), drop
    the faces that collapse and any edge shared by more than two faces, so solidify closes the sheet."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=dist)
    bad = [f for f in bm.faces if f.calc_area() < 1e-12]
    if bad:
        bmesh.ops.delete(bm, geom=bad, context="FACES")
    for _ in range(3):  # a non-manifold fan (surface nets' rare ambiguous cell): drop its faces
        nm = [e for e in bm.edges if len(e.link_faces) > 2]
        if not nm:
            break
        bmesh.ops.delete(bm, geom=list({f for e in nm for f in e.link_faces}), context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def weld_rows(post, snap=None):
    """A grid post (kit.loft_mesh) for the half-body: `post` first (on the flat vertex list), then in
    every row (a section) the run of vertices at or past the mid-plane (x <= SNAP) is moved onto the
    point where that row's surface crosses x = 0 -- the front part of the run onto the front crossing,
    the back part onto the back one, each found by interpolating between the run's end and its
    neighbour on the surface. The plain clamp (x -> 0, y kept) puts those vertices on the mid-plane
    at whatever y they had, so the weld line steps from row to row wherever the section meets the
    mid-plane at an angle (above the crotch); here it runs along the true crossing, smooth in height.
    Faces lying wholly in the run collapse and are dropped by the snap's `drop` as before."""
    from workwear import SNAP as _SNAP
    snap = _SNAP if snap is None else snap

    def f(G):
        R, C, _ = G.shape
        return np.asarray(post(G.reshape(-1, 3)), float).reshape(R, C, 3).copy()

    def g(G):
        R, C, _ = G.shape
        X = np.asarray(G, float).reshape(R, C, 3)
        raw = X[..., 0].copy()
        X = f(X)
        for r in range(R):
            run = raw[r] <= snap
            if not run.any() or run.all():
                continue
            # the run is one circular arc of columns: start where a positive column is followed by a run column
            starts = np.flatnonzero(~run & np.roll(run, -1))
            ends = np.flatnonzero(run & np.roll(~run, -1))
            if len(starts) != 1 or len(ends) != 1:
                continue
            c0, c1 = (starts[0] + 1) % C, ends[0]           # first and last run column
            p0, q0 = X[r, starts[0]], np.array([raw[r, c0], X[r, c0, 1], X[r, c0, 2]])
            p1, q1 = X[r, (c1 + 1) % C], np.array([raw[r, c1], X[r, c1, 1], X[r, c1, 2]])

            def cross(p, q):
                a, b = p[0], q[0]
                w = a / (a - b) if b < 0.0 else 1.0  # no true crossing (a grazing run): the run's end itself
                out = p + w * (q - p)
                out[0] = 0.0
                return out

            A, B = cross(p0, q0), cross(p1, q1)
            cols = (c0 + np.arange((c1 - c0) % C + 1)) % C
            front = X[r, cols, 1] < 0.5 * (A[1] + B[1])
            if A[1] > B[1]:
                A, B = B, A
            X[r, cols[front]] = A
            X[r, cols[~front]] = B
        return X

    g.grid = True
    return g


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
    if "cut" in sh:  # bounded by seam lines: the arm above the join, the torso past the raglan cut
        ph, t, dA = S.arm_coords(V)
        tj = S.t_join(ph)
        on_arm = dA < 0.0015
        ins = np.where(on_arm, t >= tj, S.in_torso_region(V))
        Q = Q[ins[Q].any(axis=1)]
        used = np.zeros(len(V), bool)
        used[Q.ravel()] = True
        out = used & ~ins
        arm_mv = out & on_arm
        V[arm_mv] = S.A.pos(ph[arm_mv], tj[arm_mv])
        z_low = S._cut_curves()[2]
        rest = np.flatnonzero(out & ~on_arm)
        if len(rest):
            p = V[rest].copy()
            low = (z_low - p[:, 2]) > (S.cut_x(p[:, 1], p[:, 2]) - p[:, 0])
            p[low, 2] = z_low
            p[~low, 0] = S.cut_x(p[~low, 1], p[~low, 2])
            fix = np.where(low[:, None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
            ee = 0.25 * h
            for _ in range(6):  # onto the union, staying on the cut's line
                f = S.union_field(p)
                g_ = np.stack([(S.union_field(p + ee * a) - S.union_field(p - ee * a)) / (2 * ee) for a in np.eye(3)], axis=1)
                g_ = g_ - np.sum(g_ * fix, axis=1)[:, None] * fix
                p = p - (f / np.maximum(np.sum(g_ * g_, axis=1), 1e-12))[:, None] * g_
            V[rest] = p
        remap = np.cumsum(used) - 1
        V, Q = V[used], remap[Q]
    elif "join" in sh:  # the arm below the join is the sleeve's: trim there, the cut exactly ON the join line
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
    V = V - (sh.get("dip", 0.0 if "cut" in sh else 0.0003) * np.clip(1.0 - dep / (2.5 * h), 0.0, 1.0))[:, None] * unit(G)
    obj = kit.new_mesh("Suit_Shoulder", V, Q, None, mat, coll)
    _clean(obj)
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
def build_suit(S, M, coll="Suit", body_only=False):
    """The suit; `body_only`: the body loft alone, no seams (a quick look at the form)."""
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
    cut = patch and "cut" in g["shoulder"]  # ... or exactly past its raglan cut (seam lines)

    def own(gr):  # the body's own displacement: its forms, or the lift onto the midplane union
        return S.mid.grid_disp(gr) if S.mid is not None else S.forms(gr.P, gr.N)

    def disp(gr):
        if cut:  # on the union near the shoulder; the shoulder's mesh owns what lies past the cut
            return own(gr) + S.edge_lift(gr.P, gr.N)
        if patch:  # sinking under the shoulder's own mesh inside its box; at the box's edge ON the union
            return own(gr) + S.edge_lift(gr.P, gr.N) - S.give_way(gr.P)[0]
        env = gr.env = F.envelope(gr.P, gr.N)
        return own(gr) + env

    armhole = g.get("armhole_seam", True)  # False: no seam round the armhole (torso and arm one surface)

    def seams(gr):
        if not armhole:
            return np.zeros(gr.P.shape[:-1])
        Pd = gr.P + getattr(gr, "env", np.zeros(gr.P.shape[:-1]))[..., None] * gr.N
        return 1.0 - np.minimum(F.armhole_seam(Pd), kit.SEAM_CAP) / kit.SEAM_CAP

    piping = g.get("seam_style", "pipe") == "tube"
    if piping or body_only:  # the seams are their own geometry (piping cords on the surface), not an attribute
        if not body_only:
            build_piping(S, body_paths, arm_paths, suit, g.get("piping", (0.0008, 0.0001)), coll)
        body_paths, arm_paths = [], []
    snap = dict(CLAMP_SNAP)
    if cut:
        snap["drop"] = lambda P: CLAMP_SNAP["drop"](P) | S.in_torso_region(P)
        snap["post"] = lambda P: CLAMP_SNAP["post"](S.body_onto_cut(P, lambda Q, Nq: S.forms(Q, Nq) + S.edge_lift(Q, Nq)))
    elif patch:
        snap["drop"] = lambda P: CLAMP_SNAP["drop"](P) | S.give_way(P)[1]
    if d.get("midline") or S.mid is not None:  # the weld exactly on the section's crossing of the mid-plane, row by row
        snap["post"] = weld_rows(snap["post"])
    rows = None
    if S.mid is not None:  # the pelvis: welded onto the union's own crossing of the plane (MidUnion)
        snap["post"], snap["drop"] = S.mid.grid_post(snap["post"]), S.mid.grid_drop(snap["drop"])
        rows = S.mid.rows(kit.RES * 0.85)
    suit_end = d["boot"].get("implicit", {}).get("suit_end")
    if suit_end:  # inside an implicit boot the suit's leg ends (its foot is the boot's)
        drop0 = snap["drop"]
        snap["drop"] = lambda P: drop0(P) | (P[:, 2] < suit_end)
    body = loft_mesh("Suit_Body", B, res=kit.RES * 0.85, rows=rows, mat=suit, coll=coll, disp=disp,
                     attrs=dict(seam=seams, pipe=lambda gr: pipe_attr(gr, body_paths)), **snap)
    if S.mid is not None:  # the run's vertices share their weld points: merge them (no zero-length edge)
        _clean(body)
        S.mid.relax(body)
    mirror(solid(body, g["cloth"], bevel=0.0), merge=True)
    if body_only:
        return body

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
            d = e if d is None else implicit.smin(d, e, im.get("foot_k", 0.003))
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
    V = implicit.taubin(V, Q, iters=im.get("smooth", 4))
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
