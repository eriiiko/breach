"""A worker in a one-piece coverall, built from a dimensions table (Blender 4.5).

Everything that differs between two workers is DATA: the dimensions table (body and
garment measures, see `worker_male/scripts/worker.py::DIMS` for the full key list), the
head and hair specs (`parts.head` / `parts.hair`) and the palette. `build(M, dims,
head_spec, hair_spec)` makes the whole figure; `materials(palette, dirt)` makes one named
material per colour region.

Conventions: the figure faces -Y, +X is its LEFT; limbs are modelled on the left and
mirrored. On a left limb `phi` reads 0 = outer side, 90 deg = front, 180 = inner side,
270 deg = back. The coverall's trunk is ONE half-body loft from the trouser hem to the
neckline, welded to its mirror image at the mid-plane (parts.CLAMP).
"""
import math

import bpy
import numpy as np
from mathutils import Matrix

import garment
import kit
import parts
import wearmat
from garment import front_phi, patch
from kit import (TAU, Loft, OnLoft, Oval, band_on, box, crumple_set, fold_field, fold_set, loft_mesh, mirror, ribbon_on, seam_attr,
                 solid, studs, tube, unit, wrap)
from parts import CLAMP, ring

D = math.radians
OUT, FRONT, IN, BACK = 0.0, D(90), D(180), D(270)
Z = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------- materials
def materials(palette, dirt=1.0):
    """One named material per colour region; every colour comes from `palette`."""
    p = palette
    return dict(
        skin=wearmat.mat_skin("worker_skin", p["SKIN"], p["HAIR"], p["LIP_TINT"]),
        hair=wearmat.mat_hair("worker_hair", p["HAIR"]),
        eye=wearmat.mat_eye("worker_eye", p["SCLERA"], p["IRIS"]),
        coverall=wearmat.mat_coverall("worker_coverall", p["COVERALL"], p["STITCH"], p["GRIME"], dirt=dirt),
        undershirt=wearmat.mat_knit("worker_undershirt", p["UNDERSHIRT"]),
        boot=wearmat.mat_leather("worker_boot", p["BOOT"], p["GRIME"], dirt=dirt),
        sole=wearmat.mat_rubber("worker_sole", p["SOLE"]),
        lace=wearmat.mat_rubber("worker_lace", p["LACE"], rough=0.85),
        metal=wearmat.mat_metal("worker_metal", p["METAL"]),
    )


def phi_where_x(loft, t, x, back=False):
    """Section angle on the front (or back) half of the loft where the surface reaches world x."""
    if not back:
        return front_phi(loft, t, x)
    ph = np.linspace(math.pi, 2.0 * math.pi, 1441)
    X = loft.pos(ph, np.full(ph.shape, t))[:, 0]
    i = np.argsort(X)
    return float(np.interp(x, X[i], ph[i]))


# --------------------------------------------------------------------- figure
class Figure:
    """The worker's surfaces, derived from one dimensions table."""

    def __init__(self, dims):
        self.d = d = dims
        self.lm = {r[0]: r[1] for r in d["trunk"]}  # landmark heights
        self.body = Loft([self._ring(*r[1:]) for r in d["trunk"]])
        self.arm = Loft([dict(p=(x, y, z), a=a, b=b, n=2.0) for _, x, y, z, a, b in d["sleeve"]])
        self.bz = self.body.t_at_z
        self.az = self.arm.t_at_z
        self.t_top = self.body.L
        self.t_elbow = self.az(d["elbow_z"])
        # the set-in sleeve's armhole: a plane through the shoulder point, leaning in towards
        # the armpit by `tilt` degrees; the sleeve owns its outer side, the body the rest
        ah = d["garment"]["armhole"]
        tl = D(ah["tilt"])
        self.arm_S = np.array([ah["x"], 0.0, ah["z"]])
        self.arm_n = np.array([math.cos(tl), 0.0, -math.sin(tl)])

    def s_arm(self, P):
        """Signed distance (m) to the armhole plane on the LEFT half: > 0 on the sleeve's side."""
        P = np.asarray(P, float)
        Pa = np.concatenate([np.abs(P[..., :1]), P[..., 1:]], axis=-1)
        return (Pa - self.arm_S) @ self.arm_n

    def _near_arm(self, P):
        return (np.abs(P[:, 0]) > 0.04) & (P[:, 2] > self.d["armpit_z"] - 0.10)

    def armhole_seam(self, P, on_sleeve=False):
        """Distance (m) to the set-in sleeve's seam: the edge of what shows of the body (or
        of the sleeve) round the armhole. Over the shoulder it is the armhole plane, where
        the two surfaces meet flush; down the front and back and under the arm it is where
        the sleeve's surface runs into the body's."""
        P = np.asarray(P, float)
        shp = P.shape[:-1]
        P = P.reshape(-1, 3)
        Pa = np.column_stack([np.abs(P[:, 0]), P[:, 1:]])
        out = np.ones(len(P))
        sel = self._near_arm(Pa)
        if sel.any():
            p = Pa[sel]
            s = self.s_arm(p)
            if on_sleeve:  # what shows of the sleeve: its own side of the plane, outside the body
                f = np.minimum(s, kit.loft_sdf(self.body, p))
            else:  # what shows of the body: its own side of the plane, or outside the sleeve
                f = np.maximum(-s, kit.loft_sdf(self.arm, p))
            out[sel] = np.abs(f)
        return out.reshape(shp)

    def envelope(self, P, N, k=0.006, under=0.002, reach=0.032):
        """How far (m, along the body's normal N) the body surface must rise to meet the
        sleeve where the sleeve stands out of it: the body takes over the sleeve's surface on
        its own side of the armhole, so the two meet flush at the seam. Beyond the armhole the
        body stays `under` below the sleeve (it is hidden there). Smooth-max fillet `k`. A point
        farther than `reach` under the sleeve's surface is the body's wall INSIDE the armhole
        (its normal runs along the sleeve, not out through it) and stays where it is."""
        P, N = np.asarray(P, float), np.asarray(N, float)
        shp = P.shape[:-1]
        P, N = P.reshape(-1, 3), N.reshape(-1, 3)
        out = np.zeros(len(P))
        sel = (P[:, 0] > 0.04) & (P[:, 2] > self.d["armpit_z"] - 0.08)
        if sel.any():
            p, n = P[sel], N[sel]
            dd = np.zeros(len(p))
            for _ in range(5):  # walk along the normal onto the sleeve's surface
                s = kit.loft_sdf(self.arm, p + dd[:, None] * n)
                dd = np.clip(dd - np.where(s < 0.5, s, 0.0), -0.02, 0.07)
            sa = self.s_arm(p)
            dd = dd * np.clip((reach - dd) / 0.008, 0.0, 1.0)
            dd = (dd - under * np.clip(sa / 0.004, 0.0, 1.0)) * np.clip(1.0 - sa / 0.012, 0.0, 1.0)
            out[sel] = kit.smax(0.0, dd, k)
        return out.reshape(shp)

    @staticmethod
    def _ring(z, cx, cy, a, bf, bb, n, level):
        r = ring(z, cx, cy, a, bf, bb, n, level if not isinstance(level, tuple) else 0)
        if isinstance(level, tuple):  # an explicit section plane (a hem higher at the front)
            r["t"] = level
        return r

    def z(self, name):
        return self.lm[name]

    def dirt(self, P):
        """Where grime collects, in world space (0..1), after the sheets: strongest on the knee
        patches and the fronts of the thighs (hands wiped on them), the forearms and cuffs,
        the chest below the pockets and the trouser hems; the back and seat only fade, faintly
        and evenly, never a stain on the centre seam. `DIMS["dirt"]` weighs each zone."""
        g = self.d["dirt"]
        P = np.asarray(P, float).reshape(-1, 3)
        x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]

        def gs(v, c, s):
            return np.exp(-((v - c) / s) ** 2)

        front = np.clip((self.d["dirt_front_y"] - y) / 0.03, 0.0, 1.0)
        legs = np.clip((self.d["crotch_z"] + 0.04 - z) / 0.04, 0.0, 1.0)
        trunk = np.clip((self.d["armpit_z"] - z) / 0.04, 0.0, 1.0) * np.clip((x - 0.0) / 0.01, 0.0, 1.0)
        knee_z = self.d["garment"]["knee"]["z"]
        chest = g["chest"] * gs(z, g["chest_z"], 0.07) * gs(x, 0.06, 0.12) * front * trunk
        knees = g["knees"] * np.exp(-np.abs((z - knee_z) / 0.085) ** 3) * front
        shins = g["shins"] * gs(z, 0.26, 0.09) * front
        thigh = g["thighs"] * gs(z, 0.68, 0.10) * front * legs
        hems = g["hems"] * np.clip(1.0 - (z - self.d["trunk"][0][1]) / 0.08, 0.0, 1.0)
        back = g["back"] * (1.0 - front) * np.clip((self.d["armpit_z"] - z) / 0.25, 0.0, 1.0) * np.clip((z - 0.25) / 0.2, 0.0, 1.0)
        # forearms: along the sleeve from the cuff to just short of the elbow
        sl = self.d["sleeve"]
        wrist, elbow = np.array(sl[0][1:4]), np.array([r for r in sl if r[0] == "elbow"][0][1:4])
        ax = elbow - wrist
        u = np.clip((np.column_stack([x, y, z]) - wrist) @ ax / (ax @ ax), -0.5, 1.5)
        near = np.linalg.norm(np.column_stack([x, y, z]) - (wrist + u[:, None] * ax), axis=1) < 0.10
        forearm = g["forearms"] * near * np.clip(1.0 - u / 0.9, 0.0, 1.0) * np.clip((u + 0.25) / 0.2, 0.0, 1.0)
        return np.clip(chest + knees + shins + thigh + hems + back + forearm, 0.0, 1.0)


# --------------------------------------------------------------------- coverall
def build_coverall(F, M, coll="Coverall"):
    d, g = F.d, F.d["garment"]
    B, A, bz, az = F.body, F.arm, F.bz, F.az
    cloth, metal = M["coverall"], M["metal"]
    dirt_g = lambda gr: F.dirt(gr.P).reshape(gr.P.shape[:2])
    z_waist, z_crotch, z_hem = g["waistband"][0], d["crotch_z"], F.z("hem")
    z_neck = B.pos([0.0], [F.t_top])[0][2]

    # authored folds: stacking on the boots, behind and over the knee, the groin, under
    # the seat, gathered at the waist and over the belly, drape under the arms
    # (heights relative to the figure's own hem, knee, crotch and armpit)
    kz, ap = g["knee"]["z"], d["armpit_z"]
    rng = np.random.default_rng(31)
    ff = fold_field(
        fold_set(rng, 16, (bz(z_hem + 0.01), bz(z_hem + 0.178)), (0, TAU), (0.0, 0.18), (0.05, 0.11), (0.006, 0.010), (0.003, 0.0065))
        + fold_set(rng, 8, (bz(kz - 0.055), bz(kz + 0.075)), (D(215), D(325)), (0.0, 0.25), (0.04, 0.08), (0.005, 0.008), (0.0025, 0.0045))
        + fold_set(rng, 6, (bz(kz - 0.035), bz(kz + 0.115)), (D(30), D(150)), (0.0, 0.25), (0.04, 0.07), (0.005, 0.008), (0.0015, 0.003))
        + fold_set(rng, 7, (bz(z_crotch - 0.165), bz(z_crotch - 0.025)), (D(30), D(160)), (-0.4, 0.3), (0.05, 0.10), (0.007, 0.011), (0.002, 0.004))
        + fold_set(rng, 5, (bz(z_crotch + 0.01), bz(z_crotch + 0.10)), (D(55), D(125)), (-0.5, 0.15), (0.05, 0.10), (0.007, 0.011), (0.003, 0.005))
        + fold_set(rng, 5, (bz(z_crotch), bz(z_crotch + 0.08)), (D(220), D(310)), (0.15, 0.2), (0.05, 0.09), (0.007, 0.011), (0.003, 0.005))
        + fold_set(rng, 10, (bz(z_waist + 0.05), bz(z_waist + 0.16)), (D(20), D(160)), (0.0, 0.2), (0.05, 0.10), (0.007, 0.011), (0.002, 0.004))
        + fold_set(rng, 8, (bz(ap - 0.07), bz(ap + 0.06)), (D(-35), D(35)), (0.6, 0.3), (0.05, 0.09), (0.006, 0.010), (0.002, 0.004))
        + crumple_set(rng, 420, (0.0, bz(z_neck - 0.06))))
    back_c = BACK - TAU  # -90 deg: the back's centre on the left half

    def elastic(gr, amp):
        """Elastic gathering across the back of the waistband: fine vertical ripples
        (`garment["elastic"]` scales it; 0 = a plain waistband)."""
        z = gr.P[..., 2]
        amp, k = amp * g.get("elastic", 1.0), g.get("elastic", 1.0)
        band = np.clip(1.0 - np.abs(z - 0.5 * (g["waistband"][0] + g["waistband"][1])) / 0.045, 0.0, 1.0)
        across = np.clip((D(62) - np.abs(wrap(gr.phi - back_c))) / D(12), 0.0, 1.0)
        return band * across * (amp * np.sin(gr.phi * g["elastic_ripples"]) - 0.007 * k)

    def fold_mask(gr):
        x, z = gr.P[..., 0], gr.P[..., 2]
        zip_zone = np.clip((np.abs(x) - 0.012) / 0.03, 0.0, 1.0)
        zip_zone = np.where((z > g["zip_bottom_z"] - 0.02) & (gr.P[..., 1] < 0.0), zip_zone, 1.0)
        return zip_zone * np.clip((z_neck - 0.03 - z) / 0.05, 0.0, 1.0)

    def disp(gr):
        hem = 0.005 * np.exp(-((gr.t - bz(z_hem) - 0.010) / 0.009) ** 2)
        env = gr.env = F.envelope(gr.P, gr.N)
        # folds die out where the body carries the sleeve's surface, so the seam stays flush
        calm = np.clip(1.0 - env / 0.002, 0.0, 1.0) * np.clip(np.abs(F.s_arm(gr.P)) / 0.03, 0.0, 1.0) ** 0.5
        calm = np.maximum(calm, np.clip((F.d["armpit_z"] - 0.06 - gr.P[..., 2]) / 0.02, 0.0, 1.0))
        return ff(gr) * fold_mask(gr) * calm + elastic(gr, 0.0040) + hem + env

    yoke = g["yoke_back_z"]

    def seams(gr):
        x, y, z = gr.P[..., 0], gr.P[..., 1], gr.P[..., 2]
        placket = np.where((y < 0.0) & (z > g["zip_bottom_z"]) & (z < z_neck), np.abs(np.abs(x) - g["placket"]), 1.0)
        fly = np.where((y < 0.0) & (z <= g["zip_bottom_z"]) & (z > z_crotch - 0.05), np.abs(x), 1.0)
        seat = np.where((y > 0.0) & (z > z_crotch - 0.05) & (z < (yoke if g.get("back_centre") else z_waist)), np.abs(x), 1.0)
        lines = [placket, fly, seat]
        for key, side in (("back_darts", y > 0.0), ("hip_pocket", y < 0.0)):  # optional stitched lines
            if key in g:
                (x0, z0), (x1, z1) = g[key]
                lines.append(np.where(side, garment.seg_dist(np.abs(x), z, (x0, z0), (x1, z1)), 1.0))
        Pd = gr.P + getattr(gr, "env", np.zeros(gr.P.shape[:-1]))[..., None] * gr.N  # where the surface really is
        armhole = F.armhole_seam(Pd)
        return seam_attr(gr, phis=[(OUT, bz(z_hem) + 0.02, bz(d["armpit_z"])), (IN, 0.0, bz(z_crotch))],
                         ts=[bz(z_hem) + 0.024, (bz(yoke), back_c, D(78))], extra=np.minimum.reduce(lines + [armhole]))

    t_top = F.t_top
    notch = Oval(B, FRONT, t_top, g["notch"][0], g["notch"][1], n=1.6, taper=0.25)
    body = loft_mesh("Coverall_Body", B, res=kit.RES * 0.85, hole=notch, mat=cloth, coll=coll, disp=disp,
                     attrs=dict(seam=seams, dirt=dirt_g), **CLAMP_SNAP)
    mirror(solid(body, g["cloth"], bevel=0.0), merge=True)

    # sleeves ---------------------------------------------------------------------
    rng = np.random.default_rng(37)
    te = F.t_elbow
    ff_arm = fold_field(
        fold_set(rng, 9, (az(g["cuff"][1]) + 0.005, az(g["cuff"][1]) + 0.10), (0, TAU), (0.0, 0.2), (0.04, 0.08), (0.006, 0.009), (0.003, 0.006))
        + fold_set(rng, 8, (te - 0.05, te + 0.04), (D(40), D(200)), (0.0, 0.25), (0.03, 0.07), (0.005, 0.009), (0.003, 0.006))
        + fold_set(rng, 5, (te - 0.03, te + 0.06), (D(240), D(330)), (0.2, 0.3), (0.03, 0.06), (0.005, 0.008), (0.002, 0.004))
        + fold_set(rng, 6, (A.t_ring(len(d["sleeve"]) - 4), A.t_ring(len(d["sleeve"]) - 2)), (D(130), D(230)), (0.5, 0.3), (0.04, 0.08), (0.006, 0.010), (0.003, 0.005))
        + crumple_set(rng, 200, (0.0, A.t_ring(len(d["sleeve"]) - 2))))
    # set in at the armhole: the sleeve ends ON the armhole plane (the body carries its surface
    # beyond, see Figure.envelope), its folds die out towards the seam, and the seam is stitched
    t_root = A.t_ring(len(d["sleeve"]) - 2)
    T_root = unit(A.pos([0.0], [A.L])[0] - A.pos([0.0], [A.L - 0.05])[0])

    def onto_armhole(P):
        s = F.s_arm(P)
        k = np.where(s < 0.0, -s / float(T_root @ F.arm_n), 0.0)
        return P + k[:, None] * T_root

    sleeve = loft_mesh("Coverall_Sleeve", A, res=kit.RES * 0.85, mat=cloth, coll=coll,
                       disp=lambda gr: ff_arm(gr) * np.clip(F.s_arm(gr.P) / 0.04, 0.0, 1.0),
                       attrs=dict(seam=lambda gr: seam_attr(gr, phis=[(IN, 0.0, t_root)], extra=F.armhole_seam(gr.P, on_sleeve=True)), dirt=dirt_g),
                       drop=lambda P: F.s_arm(P) < 0.0, post=onto_armhole)
    mirror(solid(sleeve, g["cloth"], bevel=0.0))

    # cuffs: a band with a buttoned tab on the outer-back side
    c0, c1 = az(g["cuff"][0]), az(g["cuff"][1])
    mirror(band_on("Coverall_Cuff", A, c0, c1, offset=0.0045, thick=0.004, mat=cloth, coll=coll,
                   attrs=dict(stitch=lambda gr: seam_attr(gr, ts=[c0 + 0.006, c1 - 0.006]), dirt=dirt_g)))
    tab = OnLoft(A, D(-40), 0.5 * (c0 + c1))
    mirror(patch("Coverall_Cuff_Tab", tab, 0.020, 0.5 * (c1 - c0) - 0.003, offset=0.0095, thick=0.003, n=5.0, inset=0.004,
                 attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))
    Pb, Nb = tab.pn([0.010], [0.0], 0.0095)
    mirror(studs("Coverall_Cuff_Button", Pb, Nb, r=0.0065, flat=0.35, mat=metal, coll=coll))

    # waistband: a band all round, gathered by elastic across the back
    w0, w1 = bz(g["waistband"][0]), bz(g["waistband"][1])
    band = band_on("Coverall_Waistband", B, w0, w1, offset=0.0035, thick=0.004, mat=cloth, coll=coll,
                   disp=lambda gr: elastic(gr, 0.0045), attrs=dict(stitch=lambda gr: seam_attr(gr, ts=[w0 + 0.005, w1 - 0.005]), dirt=dirt_g),
                   **CLAMP)
    mirror(band, merge=True)

    # collar ---------------------------------------------------------------------
    c = g["collar"]
    mirror(garment.shirt_collar("Coverall_Collar", B, t_top, gap_deg=c["gap"], stand=c["stand"], edge_drop=c["edge_drop"],
                                point_drop=c["point_drop"], lift=g["cloth"] + 0.002, neck=c.get("neck"), attrs=dict(dirt=F.dirt),
                                mat=cloth, coll=coll), merge=True)

    # front zip ---------------------------------------------------------------------
    z_zip_top = z_neck - g["notch"][1] + 0.004
    zs = np.linspace(z_zip_top, g["zip_bottom_z"], 40)
    path = [(math.degrees(front_phi(B, bz(z))), bz(z)) for z in zs]
    ribbon_on("Coverall_Zip_Tape", B, path, 0.016, offset=0.0035, thick=0.0025, mat=cloth, coll=coll)
    ribbon_on("Coverall_Zip_Teeth", B, path, 0.0065, offset=0.0052, thick=0.0025, mat=metal, coll=coll)
    zip_slider(B, front_phi(B, bz(z_zip_top - 0.012)), bz(z_zip_top - 0.012), 0.0075, metal, coll, "Coverall_Zip_Pull")

    # pockets ------------------------------------------------------------------------
    cp = g["chest_pocket"]
    t = bz(cp["z"])
    ph = front_phi(B, t, cp["x"])
    pk = OnLoft(B, ph, t)
    r = float(B.radius(ph, t)[0])

    def zip_pocket(sfx):
        """A patch pocket with a zip across its top."""
        objs = [patch("Coverall_Chest_Pocket" + sfx, pk, cp["hs"], cp["ht"], offset=0.0050, thick=0.0035, n=9.0, attrs=dict(dirt=F.dirt),
                      mat=cloth, coll=coll)]
        zt = t + cp["ht"] - cp["zip_dt"]
        dph = math.degrees((cp["hs"] - 0.010) / r)
        zpath = [(math.degrees(ph) + dph, zt), (math.degrees(ph) - dph, zt)]
        objs += [ribbon_on("Coverall_Chest_Zip_Tape" + sfx, B, zpath, 0.012, offset=0.0062, thick=0.002, mat=cloth, coll=coll),
                 ribbon_on("Coverall_Chest_Zip_Teeth" + sfx, B, zpath, 0.0045, offset=0.0074, thick=0.002, mat=metal, coll=coll)]
        objs += zip_slider(B, ph + math.radians(dph) - 0.012 / r, zt - 0.008, 0.0078, metal, coll, "Coverall_Chest_Zip_Pull" + sfx)
        return objs

    def flap_pocket(sfx):
        """A patch pocket with a buttoned flap over its top and a stitched pen slot."""
        objs = [patch("Coverall_Chest_Pocket" + sfx, pk, cp["hs"], cp["ht"], offset=0.0050, thick=0.0035, n=9.0, attrs=dict(dirt=F.dirt),
                      mat=cloth, coll=coll)]
        fh = cp.get("flap_h", 0.024)
        objs.append(patch("Coverall_Chest_Flap" + sfx, pk, cp["hs"] + 0.004, fh * 0.5, shift=(0.0, cp["ht"] - fh * 0.5 + 0.004), offset=0.0080,
                          thick=0.0035, n=8.0, inset=0.004, point=cp.get("flap_point", 0.006), attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))
        P, N = pk.pn([0.0], [cp["ht"] - fh + 0.004 - cp.get("flap_point", 0.006) + 0.007], 0.0082)
        objs.append(studs("Coverall_Chest_Button" + sfx, P, N, r=0.0055, flat=0.40, mat=metal, coll=coll))
        if cp.get("pen_slot"):
            objs.append(patch("Coverall_Chest_Pen_Slot" + sfx, pk, 0.005, cp["ht"] - 0.5 * fh - 0.008, shift=(cp["pen_slot"], -0.5 * fh - 0.002),
                              offset=0.0068, thick=0.002, n=6.0, inset=0.002, attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))
        return objs

    kinds = dict(zip=zip_pocket, flap=flap_pocket)
    left, right = cp.get("left", "zip"), cp.get("right", "zip")
    if left == right:
        for o in kinds[left](""):
            mirror(o)
    else:  # a different pocket on each breast: build both on the left, move one across
        kinds[left]("_L")
        for o in kinds[right]("_R"):
            kit.flip_x(o)

    sp = g.get("sleeve_pocket")  # left arm only, where the worker has one
    if sp:
        ta = az(sp["z"])
        spk = OnLoft(A, D(sp["phi"]), ta)
        patch("Coverall_Sleeve_Pocket", spk, sp["hs"], sp["ht"], offset=0.0040, thick=0.003, n=9.0, attrs=dict(dirt=F.dirt), mat=cloth, coll=coll)
        patch("Coverall_Sleeve_Pocket_Flap", spk, sp["hs"] + 0.003, 0.014, shift=(0.0, sp["ht"] - 0.010), offset=0.0072, thick=0.003, n=9.0,
              inset=0.004, attrs=dict(dirt=F.dirt), mat=cloth, coll=coll)
        patch("Coverall_Sleeve_Pen_Slot", spk, 0.006, sp["ht"] * 0.55, shift=(sp["hs"] * 0.62, -0.006), offset=0.0068, thick=0.002, n=6.0,
              inset=0.002, attrs=dict(dirt=F.dirt), mat=cloth, coll=coll)

    for i, (x, back) in enumerate(g.get("belt_loops", ())):  # short tabs stitched over the waistband
        tb = 0.5 * (w0 + w1)
        lp = OnLoft(B, phi_where_x(B, tb, x, back=back), tb)
        mirror(patch("Coverall_Belt_Loop_%d" % i, lp, 0.006, 0.5 * (w1 - w0) + 0.006, offset=0.0085, thick=0.003, n=5.0, inset=0.0025,
                     attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))

    bp = g["back_pocket"]
    t = bz(bp["z"])
    pk = OnLoft(B, phi_where_x(B, t, bp["x"], back=True), t)
    mirror(patch("Coverall_Back_Pocket", pk, bp["hs"], bp["ht"], offset=0.0045, thick=0.0035, n=8.0, lines=(bp.get("hem_y", -(bp["ht"] - 0.024)),),
                 point=bp.get("point", 0.0), attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))

    cg = g["cargo"]
    t = bz(cg["z"])
    pk = OnLoft(B, D(cg["phi"]), t)
    mirror(patch("Coverall_Cargo_Pocket", pk, cg["hs"], cg["ht"], offset=cg["depth"], thick=cg["depth"] - 0.003, n=10.0, bevel=0.0045, seg=3,
                 attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))
    fl = OnLoft(B, D(cg["phi"]), t + cg["ht"] - 0.006)
    mirror(patch("Coverall_Cargo_Flap", fl, cg["hs"] + 0.004, 0.026, offset=cg["depth"] + 0.0055, thick=0.005, n=8.0, inset=0.0045,
                 attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))
    if cg.get("snaps", True):
        P, N = fl.pn([-0.035, 0.035], [-0.010, -0.010], cg["depth"] + 0.0055)
        mirror(studs("Coverall_Cargo_Snap", P, N, r=0.0055, flat=0.35, mat=metal, coll=coll))

    kp = g["knee"]
    mirror(patch("Coverall_Knee_Patch", OnLoft(B, FRONT, bz(kp["z"])), kp["hs"], kp["ht"], offset=0.0050, thick=0.0035, n=4.0, inset=0.006,
                 attrs=dict(dirt=F.dirt), mat=cloth, coll=coll))


# The half-body weld with a snap band: where the clamped surface meets the mid-plane at a
# shallow angle (the crotch, the seat) its last vertices sit a millimetre or two off the
# plane and the mirror merge would leave a slit. Vertices within SNAP of the plane go onto
# it, and faces lying wholly in that band are dropped, so the two halves meet edge to edge.
SNAP = 0.003
CLAMP_SNAP = dict(drop=lambda P: P[:, 0] <= SNAP,
                  post=lambda P: np.column_stack([np.where(P[:, 0] > SNAP, P[:, 0], 0.0), P[:, 1:]]))


def zip_slider(loft, phi, t, off, mat, coll, name, mirrored=False):
    """A zip's slider and its hanging pull tab, sitting on a loft."""
    a = OnLoft(loft, phi, t)
    Fm = kit.anchor_matrix(a, lift=off)
    objs = [box(name, (0.010, 0.012, 0.005), M=Fm @ Matrix.Translation((0, 0.001, 0.0025)), bevel=0.0015, seg=2, mat=mat, coll=coll),
            box(name + "_Tab", (0.008, 0.020, 0.0018), M=Fm @ Matrix.Translation((0, -0.013, 0.0048)) @ Matrix.Rotation(0.15, 4, "X"),
                bevel=0.0008, seg=2, mat=mat, coll=coll)]
    if mirrored:
        for o in objs:
            mirror(o)
    return objs


def build_undershirt(F, M, coll="Coverall"):
    """The dark crew-neck undershirt that shows in the open neck of the coverall."""
    u = F.d["undershirt"]
    lo = Loft([ring(z, 0.0, cy, a, bf, bb, n, 1) for z, cy, a, bf, bb, n in u["rings"]])
    solid(loft_mesh("Undershirt", lo, res=kit.RES * 0.8, mat=M["undershirt"], coll=coll), 0.002, bevel=0.0)
    t1 = lo.L
    band_on("Undershirt_Rib", lo, t1 - u["rib"], t1, offset=0.0015, thick=0.003, mat=M["undershirt"], coll=coll)


# ------------------------------------------------------------------------- hands
def build_hands(F, M, coll="Hands"):
    h = F.d["hand"]
    w = np.array(F.d["sleeve"][0][1:4], float)
    L = unit(h["down"])
    b = np.asarray(h["back"], float)
    Bv = unit(b - (b @ L) * L)
    wrist = w + L * h["drop"]
    parts.hand("Hand", wrist, L, Bv, M["skin"], coll, scale=h["scale"], curl=h["curl"], girth=h["girth"], palm_girth=h["palm_girth"],
               palm=h["palm"])
    garment.close_holes([o for o in bpy.data.objects if o.name.startswith("Hand_")])


# ------------------------------------------------------------------------- boots
class Boot:
    def __init__(self, spec):
        self.s = s = spec
        c, sn = math.cos(D(s["toe_out"])), math.sin(D(s["toe_out"]))
        self.BX, self.BY = np.array([c, sn, 0.0]), np.array([-sn, c, 0.0])
        self.ankle = np.array([s["ankle"][0], s["ankle"][1], 0.0])
        self.y_toe, self.y_heel = s["profile"][0][0], s["profile"][-1][0]

    def w(self, x, y, z):
        return self.ankle + x * self.BX + y * self.BY + z * Z

    def sole_top(self, y):
        s = self.s
        return s["sole"][0] + (s["sole"][1] - s["sole"][0]) * (y - self.y_toe) / (self.y_heel - self.y_toe)


def build_boots(F, M, coll="Boots"):
    bt = Boot(F.d["boot"])
    s, w, st = bt.s, bt.w, bt.sole_top
    prof = s["profile"]
    leather, rub = M["boot"], M["sole"]
    ft = lambda y: y - bt.y_toe
    sole = Loft([dict(p=w(0, y, st(y) / 2), a=a + s["welt"], bf=st(y) / 2 + 0.004, bb=st(y) / 2, n=7.0) for y, a, _ in prof], front=Z)
    mirror(loft_mesh("Boot_Sole", sole, res=0.003, disp=parts.tread(bt.y_toe, pitch=0.024, lug=0.004, arch_y=-0.03, arch_w=0.05),
                     cap0=True, cap1=True, mat=rub, coll=coll))
    welt = Loft([dict(p=w(0, y, st(y) + 0.004), a=a + s["welt"] - 0.001, bf=0.006, bb=0.006, n=6.0) for y, a, _ in prof], front=Z)
    mirror(loft_mesh("Boot_Welt", welt, res=0.0028, cap0=True, cap1=True, mat=leather, coll=coll,
                     attrs=dict(stitch=lambda g: seam_attr(g, phis=[D(60), D(120)]))))
    foot = Loft([dict(p=w(0, y, st(y) + 0.008), a=a, bf=h, bb=0.012, n=(2.5, 7.0)) for y, a, h in prof], front=Z)
    mirror(loft_mesh("Boot_Upper", foot, res=0.003, cap0=True, cap1=True, mat=leather, coll=coll,
                     attrs=dict(seam=lambda g: seam_attr(g, ts=[ft(s["toe_cap_y"]), ft(s["vamp_y"])]), dirt=lambda g: np.full(g.phi.shape, 0.6))))
    shaft = Loft([dict(p=w(0, s["shaft_y"], z), a=a, bf=bf, bb=bb, n=2.2) for z, a, bf, bb in s["shaft"]], front=-bt.BY)
    mirror(loft_mesh("Boot_Shaft", shaft, res=0.003, mat=leather, coll=coll, cap0=True, cap1=True,
                     attrs=dict(seam=lambda g: seam_attr(g, phis=[BACK - TAU + 0.0, D(20), D(160)]))))
    sz = shaft.t_at_z
    mirror(band_on("Boot_Collar", shaft, shaft.L - 0.022, shaft.L, offset=0.004, thick=0.006, mat=leather, coll=coll))
    top = lambda y: OnLoft(foot, FRONT, ft(y))
    mirror(patch("Boot_Toe_Cap", top(s["toe_cap_y"] - 0.036), 0.060, 0.040, offset=0.0025, thick=0.003, n=3.0, inset=0.004,
                 mat=leather, coll=coll))
    mirror(patch("Boot_Heel_Counter", OnLoft(shaft, BACK, sz(0.078)), 0.055, 0.026, offset=0.003, thick=0.003, n=4.0, inset=0.004,
                 mat=leather, coll=coll))
    # lacing on the instep: a tongue between two raised facings, eyelets, crossed laces
    y0, y1 = s["vamp_y"], s["lace_top_y"]
    ym = 0.5 * (y0 + y1)
    inst = OnLoft(foot, FRONT, ft(ym))
    half = 0.5 * (y1 - y0)
    mirror(patch("Boot_Tongue", inst, s["lace_half"] + 0.004, half + 0.012, shift=(0.0, 0.004), offset=0.0035, thick=0.003, n=4.0,
                 inset=0.0035, mat=leather, coll=coll))
    for sg in (-1, 1):
        mirror(patch("Boot_Facing", inst, 0.010, half + 0.006, shift=(sg * (s["lace_half"] + 0.006), 0.0), offset=0.0070, thick=0.004,
                     n=6.0, inset=0.003, mat=leather, coll=coll))
    rows = np.linspace(-half + 0.008, half - 0.004, s["lace_rows"])
    E, En = [], []
    for tt in rows:
        P, N = inst.pn([-s["lace_half"], s["lace_half"]], [tt, tt], 0.0075)
        E += list(P)
        En += list(N)
    mirror(studs("Boot_Eyelets", np.array(E), np.array(En), r=0.0042, flat=0.45, mat=M["metal"], coll=coll))
    k = 0
    for i in range(len(rows) - 1):
        for (sa, ta), (sb, tb) in (((-1, rows[i]), (1, rows[i + 1])), ((1, rows[i]), (-1, rows[i + 1]))):
            u = np.linspace(0.0, 1.0, 12)
            ss = s["lace_half"] * (sa + (sb - sa) * u)
            tt = ta + (tb - ta) * u
            P, _ = inst.pn(ss, tt, 0.0090 + 0.0025 * np.sin(np.pi * u) + (0.0018 if sa < 0 else 0.0))
            mirror(tube("Boot_Lace_%d" % k, P, r=0.0021, n_u=8, mat=M["lace"], coll=coll))
            k += 1


# -------------------------------------------------------------------------- head
def build_head(F, M, head_spec, hair_spec, coll="Head"):
    loft, _ = parts.head("Worker", head_spec, dict(skin=M["skin"], eye=M["eye"]), coll=coll)
    parts.hair("Worker", loft, hair_spec, M["hair"], coll=coll)
    if "bun" in hair_spec:
        parts.hair_bun("Worker", hair_spec["bun"], M["hair"], coll=coll)
    if "strands" in hair_spec:
        parts.hair_strands("Worker", loft, hair_spec["strands"], M["hair"], coll=coll)
    return loft


def build(M, dims, head_spec, hair_spec):
    F = Figure(dims)
    build_coverall(F, M)
    build_undershirt(F, M)
    build_hands(F, M)
    build_boots(F, M)
    build_head(F, M, head_spec, hair_spec)
    garment.close_holes([o for o in bpy.data.objects if o.name.startswith(("Boot_Lace", "Boot_Eyelets", "Coverall_Cargo_Snap", "Coverall_Cuff_Button", "Coverall_Chest_Button"))])
    return F
