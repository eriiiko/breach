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
import kit
import parts
import wearmat
from garment import front_phi
from kit import Loft, OnLoft, band_on, box, loft_mesh, mirror, ribbon_on, seam_attr, solid, tube, unit
from workwear import CLAMP_SNAP, Figure, build_hands, zip_slider

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
        mesh=wearmat.mat_mesh(prefix + "_mesh", p["MESH"], p["SKIN"], cell=g.get("mesh_cell", 0.0032), show=g.get("mesh_show", 0.55)),
        skin=wearmat.mat_skin(prefix + "_skin", p["SKIN"], p["BROW"], p["LIP_TINT"]),
        eye=wearmat.mat_eye(prefix + "_eye", p["SCLERA"], p["IRIS"]),
        boot=wearmat.mat_gloss(prefix + "_boot", p["BOOT"], rough=g.get("boot_rough", 0.20), coat=g.get("boot_coat", 0.8),
                               coat_rough=g.get("boot_coat_rough", 0.08), piping=0.5),
        sole=wearmat.mat_rubber(prefix + "_sole", p["SOLE"], rough=0.55),
        metal=wearmat.mat_metal(prefix + "_metal", p["METAL"], rough=0.25),
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
        self.B, self.A, self.bz, self.az = self.F.body, self.F.arm, self.F.bz, self.F.az
        self.frame = Frame(dims["trunk"])
        self.ref = Frame(dims.get("feature_frame", dims["trunk"]))

    # --- points --------------------------------------------------------------
    def phi_t(self, pt, on="body"):
        """A feature point -> (phi radians, t) on the body (or arm) loft."""
        kind, v, z = pt
        if on == "arm":
            return D(v), self.az(z)
        t = self.bz(z)
        if kind == "phi":
            return D(v), t
        return front_phi(self.B, t, self.frame.remap(v, z, self.ref)), t

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
        return self.forms(P, N) + self.F.envelope(P, N)


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

    def disp(gr):
        env = gr.env = F.envelope(gr.P, gr.N)
        return S.forms(gr.P, gr.N) + env

    def seams(gr):
        Pd = gr.P + getattr(gr, "env", np.zeros(gr.P.shape[:-1]))[..., None] * gr.N
        return 1.0 - np.minimum(F.armhole_seam(Pd), kit.SEAM_CAP) / kit.SEAM_CAP

    body = loft_mesh("Suit_Body", B, res=kit.RES * 0.85, mat=suit, coll=coll, disp=disp,
                     attrs=dict(seam=seams, pipe=lambda gr: pipe_attr(gr, body_paths)), **CLAMP_SNAP)
    mirror(solid(body, g["cloth"], bevel=0.0), merge=True)

    # the sleeve, set in at the armhole (as the coverall's), smooth
    T_root = unit(A.pos([0.0], [A.L])[0] - A.pos([0.0], [A.L - 0.05])[0])

    def onto_armhole(P):
        s = F.s_arm(P)
        k = np.where(s < 0.0, -s / float(T_root @ F.arm_n), 0.0)
        return P + k[:, None] * T_root

    def sleeve_seams(gr):
        return 1.0 - np.minimum(F.armhole_seam(gr.P, on_sleeve=True), kit.SEAM_CAP) / kit.SEAM_CAP

    sleeve = loft_mesh("Suit_Sleeve", A, res=kit.RES * 0.85, mat=suit, coll=coll,
                       attrs=dict(seam=sleeve_seams, pipe=lambda gr: pipe_attr(gr, arm_paths)),
                       drop=lambda P: F.s_arm(P) < 0.0, post=onto_armhole)
    mirror(solid(sleeve, g["cloth"], bevel=0.0))

    # cuff bands at the wrists
    c0, c1 = az(g["cuff"][0]), az(g["cuff"][1])
    mirror(band_on("Suit_Cuff", A, c0, c1, offset=0.0010, thick=0.0025, mat=suit, coll=coll, bevel=0.0008,
                   attrs=dict(seam=lambda gr: seam_attr(gr, ts=[c0 + 0.0018, c1 - 0.0018]))))

    # the stand collar: a tube round the neck, rising from the neckline to under the jaw
    col = g["collar"]
    CL = Loft([parts.ring(z, 0.0, cy, a, bf, bb, n, 1) for z, cy, a, bf, bb, n in col["rings"]])
    collar = loft_mesh("Suit_Collar", CL, res=kit.RES * 0.7, mat=suit, coll=coll,
                       attrs=dict(seam=lambda gr: seam_attr(gr, ts=[CL.L - col.get("top_seam", 0.006)])))
    solid(collar, col.get("thick", 0.003), bevel=0.0012)

    # the front zip: teeth from the collar's top down the front to below the navel, the pull at the top
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

    # mesh insert panels, each framed by a piping tube
    for i, pn in enumerate(g["mesh_panels"]):
        on = pn.get("on", "body")
        lo = A if on == "arm" else B
        outline = [S.phi_t(p, on) for p in pn["pts"]]
        name = "Suit_Mesh_%s" % pn.get("name", i)
        obj, rim = garment.panel(name, lo, outline, offset=0.0006, thick=0.0012, lift=None if on == "arm" else S.lift, mat=M["mesh"],
                                 coll=coll)
        mirror(obj)
        mirror(tube(name + "_Piping", rim, r=pn.get("piping", 0.0012), closed=True, n_u=8, mat=suit, coll=coll))

    # shaped knee pads: a domed panel over the knee cap
    kp = g["knee_pad"]
    ph, t = S.phi_t(("x", kp["x"], kp["z"]))
    mirror(garment.patch("Suit_Knee_Pad", OnLoft(B, ph, t), kp["hs"], kp["ht"], offset=kp.get("offset", 0.0030), thick=0.0035,
                         n=kp.get("n", 2.4), dome=kp.get("dome", 0.003), inset=kp.get("inset", 0.004), mat=suit, coll=coll))


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
    shaft_ob = band_on("Boot_Shaft", B, t0, t1, offset=s["shaft_ease"], thick=0.0025, bevel=0.0010, mat=boot, coll=coll, post=v_top,
                       attrs=dict(seam=lambda g: seam_attr(g, ts=[t1 - 0.004], phis=[(FRONT + D(s["panel_phi"]), t0, bz(z1 - vd)),
                                                                                     (FRONT - D(s["panel_phi"]), t0, bz(z1 - vd))])))
    mirror(shaft_ob)


# ------------------------------------------------------------------------- head
def build_head(M, head_spec, prefix, coll="Head"):
    """The bald placeholder head: `parts.head` with this figure's spec. Its own collection and
    objects (`<prefix>_Head`, `_Eye_*`, `_Ear`), so a later stage can replace it whole."""
    loft, _ = parts.head(prefix, head_spec, dict(skin=M["skin"], eye=M["eye"]), coll=coll)
    return loft


def build(M, dims, head_spec, prefix):
    S = Suit(dims)
    build_suit(S, M)
    build_hands(S.F, M)
    build_boots(S, M)
    build_head(M, head_spec, prefix)
    garment.close_holes([o for o in bpy.data.objects if o.name.startswith(("Suit_Zip_Stop",))])
    return S
