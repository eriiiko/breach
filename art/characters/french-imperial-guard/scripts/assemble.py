"""Assembly of the guard officer from his tables (`guard.py`) on the kit (Blender 4.5).

One-off placement code for this uniform; the reusable pieces live in `charkit/`:
`workwear.Figure` (the body loft, the set-in sleeve, its envelope), `dresswear` (free-edged
loft regions, straps across the welded half-body, swept tubes, the riding boot and spur),
`braid` (lace as a `braid` attribute, cords, tassels), `fur` (fur rolls, plumes), `shako`,
`dressmat` (the materials), `parts` (head, hair, hands).

`Builder(tables, variant)` is what `charkit/workerbuild.run` drives: `materials(palette,
dirt)` and `build(M, dims, head, hair)`.
"""
import math

import bpy
import numpy as np

import braid
import dressmat
import dresswear
import fur
import garment
import kit
import parts
import shako
import wearmat
import workwear
from braid import line_value, polyline_dist
from dresswear import half_body_pn, loft_region, smooth_path, strap, swept
from kit import TAU, Loft, OnLoft, band_on, crumple_set, fold_field, fold_set, loft_mesh, mirror, solid, studs, unit, wrap
from workwear import CLAMP_SNAP

D = math.radians
Z = np.array([0.0, 0.0, 1.0])


class Builder:
    def __init__(self, tables, variant=None):
        self.T, self.variant = tables, variant or None

    def materials(self, palette, dirt=0.0):
        return materials(palette, self.T.plumes(self.variant))

    def build(self, M, dims, head, hair):
        return build(M, dims, head, hair, self.T.plumes(self.variant))


# ------------------------------------------------------------------- materials
def materials(p, plumes):
    """One named material per region (`guard_*`); every colour from the palette `p`."""
    M = dict(
        skin=wearmat.mat_skin("guard_skin", p["SKIN"], p["HAIR"], p["LIP_TINT"]),
        hair=wearmat.mat_hair("guard_hair", p["HAIR"]),
        eye=wearmat.mat_eye("guard_eye", p["SCLERA"], p["IRIS"]),
        dolman=dressmat.mat_wool("guard_dolman", p["DOLMAN"], lace=p["GOLD"], relief=3.5, relief_d=0.0035),
        pelisse=dressmat.mat_wool("guard_pelisse", p["PELISSE"], lace=p["GOLD"], relief=3.5, relief_d=0.0035),
        cuff=dressmat.mat_wool("guard_cuff", p["CUFF"], lace=p["GOLD"]),
        breeches=dressmat.mat_wool("guard_breeches", p["BREECHES"], lace=p["GOLD"]),
        fur=dressmat.mat_fur("guard_fur", p["FUR"], p["FUR_TIP"]),
        braid=dressmat.mat_gold_cord("guard_braid", p["GOLD"]),
        metal=dressmat.mat_gold_metal("guard_metal", p["METAL"]),
        steel=wearmat.mat_metal("guard_steel", p["STEEL"], rough=0.30),
        sash=dressmat.mat_sash("guard_sash", p["SASH"], p["SASH_STRIPE"], p["GOLD"]),
        boot=dressmat.mat_polished("guard_boot", p["BOOT"], rough=0.30, coat=0.35, coat_rough=0.24, crease=0.7),
        sole=wearmat.mat_rubber("guard_sole", p["SOLE"], rough=0.6),
        glove=dressmat.mat_glove("guard_glove", p["GLOVE"]),
        shako=dressmat.mat_shako("guard_shako", p["SHAKO"]),
        peak=dressmat.mat_polished("guard_peak", p["PEAK"], rough=0.12),
        cockade=dressmat.mat_wool("guard_cockade", p["COCKADE"], rough=0.6),
    )
    for i, pl in enumerate(plumes):
        M["plume_%d" % i] = dressmat.mat_feather("guard_plume_" + "abcd"[i], p[pl["colour"]], p[pl["tip"]] if pl.get("tip") else None,
                                                 pl.get("tip_from", 0.75), blend=pl.get("tip_blend"))
    return M


# ------------------------------------------------------------------ lace layout
def front_back(gr_P, front_lines, back_lines, side=0.01):
    """Distance to lace drawn in the front (x, z) projection on the front of the body, and in
    the back projection on its back (both in signed x; a half-body uses |x|)."""
    P = gr_P.reshape(-1, 3)
    Q = np.column_stack([np.abs(P[:, 0]), P[:, 2]])
    d = np.ones(len(P))
    if front_lines:
        df = polyline_dist(Q, front_lines)
        d = np.where(P[:, 1] < side, np.minimum(d, df), d)
    if back_lines:
        db = polyline_dist(Q, back_lines)
        d = np.where(P[:, 1] > -side, np.minimum(d, db), d)
    return d.reshape(gr_P.shape[:-1])


def bold_v(s0, t0, half, drop, n=3, pitch=0.004):
    """A bold chevron (an inverted V of wide lace) in developed (s, t) coordinates: `n` parallel
    Vs `pitch` apart, so the lace bands merge into one wide band."""
    out = []
    for k in range(n):
        dt = (k - 0.5 * (n - 1)) * pitch
        out.append(np.array([[s0 - half, t0 - drop + dt], [s0, t0 + dt], [s0 + half, t0 - drop + dt]]))
    return out


def dolman_lace(F):
    g = F.d["garment"]
    fr = g["frogs"]
    rows = braid.frog_rows(fr["z0"], fr["z1"], fr["n"], lambda z: fr["w0"] + (fr["w1"] - fr["w0"]) * (z - fr["z0"]) / (fr["z1"] - fr["z0"]),
                           loop=fr["loop"])
    bk = g["back"]
    curve = smooth_path([[x, 0.0, z] for x, z in bk["curve"]], 30)[:, [0, 2]]
    back = [curve, np.array([[0.0, bk["centre"][0]], [0.0, bk["centre"][1]]])]
    for x, z in (bk["curve"][0], bk["curve"][-1]):
        back += braid.trefoil((x, z), bk["knot"], up=1.0 if z > 1.3 else -1.0)
    back += braid.trefoil((0.0, bk["centre"][1]), bk["knot"], up=-1.0)
    hem = g["dolman_hem"] + 0.0045
    sc = np.array(g["shoulder_cord"])

    def f(gr):
        P = gr.P
        d = front_back(P, rows, back)
        d = np.minimum(d, np.abs(P[..., 2] - hem))  # the hem's edging all round
        # the shoulder cord along the top of each shoulder (seen from above: x, y near the top)
        Q = np.column_stack([np.abs(P[..., 0]).ravel(), P[..., 2].ravel()])
        ds = polyline_dist(Q, [sc]).reshape(P.shape[:-1])
        top = np.abs(P[..., 1] - 0.012) < 0.035
        d = np.where(top, np.minimum(d, ds), d)
        return line_value(d)

    return f


def offset_lines(lines, d):
    """Each (x, z) polyline doubled: two copies `d` either side along its own normal (0: as is)."""
    if not d:
        return lines
    out = []
    for l in lines:
        l = np.asarray(l, float)
        T = np.gradient(l, axis=0)
        T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
        Nn = np.column_stack([-T[:, 1], T[:, 0]])
        out += [l + d * Nn, l - d * Nn]
    return out


def breeches_lace(F, B):
    g = F.d["garment"]
    kn = g["knot"]
    # the Hungarian knot in heavy braid: each line doubled, two cords side by side
    knot = offset_lines(braid.hungarian_knot((kn["x"], kn["top"]), kn["height"], kn["width"]), kn.get("double", 0.0))
    se = g["seat"]
    seat = [smooth_path([[x, 0.0, z] for x, z in se["curve"]], 30)[:, [0, 2]]]
    seat += braid.trefoil(se["curve"][0], se["knot"], up=-1.0)
    gap = g["stripe"]["gap"]
    t_knee = B.t_at_z(0.36)

    def f(gr):
        d = front_back(gr.P, knot, seat)
        s = np.abs(wrap(gr.phi)) * gr.r  # distance round from the outer seam (phi 0)
        stripe = np.abs(s - gap)
        d = np.minimum(d, np.where(gr.t >= t_knee, stripe, 1.0))
        return line_value(d)

    return f


# ------------------------------------------------------------------------- body
def build_body(F, M, coll="Uniform"):
    d, g = F.d, F.d["garment"]
    B, A, bz, az = F.body, F.arm, F.bz, F.az
    rng = np.random.default_rng(41)
    z_hem, z_btop = g["dolman_hem"], g["breeches_top"]
    kz = 0.50
    # breeches: knee and seat folds over a faint crumple (a tight-fitting cut)
    # tight cloth: knee and groin folds only, no all-over crumple
    ff_b = fold_field(fold_set(rng, 6, (bz(kz - 0.04), bz(kz + 0.06)), (D(215), D(325)), (0.0, 0.25), (0.03, 0.06), (0.004, 0.007), (0.0015, 0.003))
                      + fold_set(rng, 5, (bz(0.75), bz(0.84)), (D(40), D(140)), (-0.4, 0.3), (0.04, 0.08), (0.006, 0.010), (0.0015, 0.003)))
    breeches = loft_mesh("Breeches", B, t0=0.0, t1=bz(z_btop), res=kit.RES * 0.85, mat=M["breeches"], coll=coll,
                         disp=lambda gr: 0.5 * ff_b(gr), attrs=dict(braid=breeches_lace(F, B)), **CLAMP_SNAP)
    mirror(solid(breeches, g["cloth"], bevel=0.0), merge=True)

    # dolman body: from its hem to the neckline, rising onto the set-in sleeves
    ff_d = fold_field(fold_set(rng, 8, (bz(d["armpit_z"] - 0.07), bz(d["armpit_z"] + 0.05)), (D(-35), D(35)), (0.6, 0.3), (0.05, 0.08), (0.006, 0.009), (0.0015, 0.003))
                      + crumple_set(rng, 160, (bz(z_hem), bz(1.45))))
    t_hem = bz(z_hem)

    def dolman_off(gr):
        return 0.0045 + 0.004 * np.clip(1.0 - (gr.t - t_hem) / 0.05, 0.0, 1.0)

    def dolman_disp(gr):
        env = F.envelope(gr.P, gr.N)
        calm = np.clip(1.0 - env / 0.002, 0.0, 1.0) * np.clip(np.abs(F.s_arm(gr.P)) / 0.03, 0.0, 1.0) ** 0.5
        calm = np.maximum(calm, np.clip((d["armpit_z"] - 0.06 - gr.P[..., 2]) / 0.02, 0.0, 1.0))
        return 0.6 * ff_d(gr) * calm + env

    body = loft_mesh("Dolman_Body", B, t0=t_hem, t1=B.L, res=kit.RES * 0.85, offset=dolman_off, disp=dolman_disp, mat=M["dolman"], coll=coll,
                     attrs=dict(braid=dolman_lace(F)), **CLAMP_SNAP)
    mirror(solid(body, g["cloth"], bevel=0.0), merge=True)

    # dolman sleeves, set in at the armhole
    rng = np.random.default_rng(43)
    te = F.t_elbow
    ff_a = fold_field(fold_set(rng, 7, (te - 0.05, te + 0.04), (D(40), D(200)), (0.0, 0.25), (0.03, 0.06), (0.005, 0.008), (0.002, 0.004))
                      + crumple_set(rng, 120, (0.0, A.t_ring(len(d["sleeve"]) - 2))))
    T_root = unit(A.pos([0.0], [A.L])[0] - A.pos([0.0], [A.L - 0.05])[0])

    def onto_armhole(P):
        s = F.s_arm(P)
        k = np.where(s < 0.0, -s / float(T_root @ F.arm_n), 0.0)
        return P + k[:, None] * T_root

    cf = g["cuff"]
    t_cuff = az(cf["top"])
    pp = D(cf.get("point_phi", 0.0))  # where round the arm the point stands (0 outside, 90 deg front)
    cuff_top = lambda phi: t_cuff + cf["point"] * np.clip(1.0 - np.abs(wrap(phi - pp)) / D(cf["point_w"]), 0.0, 1.0)
    ch = g["chevrons"]
    r_c = float(A.radius([0.0], [t_cuff])[0])
    chev = []
    for i in range(ch["n"]):
        tp = t_cuff + cf["point"] + (i + 1) * ch["gap"]
        chev += bold_v(0.0, tp, ch["half"], ch["drop"], ch.get("bold", 1), ch.get("pitch", 0.004))

    def sleeve_lace(gr):
        Q = np.column_stack([(wrap(gr.phi - pp) * gr.r).ravel(), gr.t.ravel()])
        return line_value(polyline_dist(Q, chev).reshape(gr.phi.shape))

    sleeve = loft_mesh("Dolman_Sleeve", A, res=kit.RES * 0.85, mat=M["dolman"], coll=coll,
                       disp=lambda gr: 0.6 * ff_a(gr) * np.clip(F.s_arm(gr.P) / 0.04, 0.0, 1.0), attrs=dict(braid=sleeve_lace),
                       drop=lambda P: F.s_arm(P) < 0.0, post=onto_armhole)
    mirror(solid(sleeve, g["cloth"], bevel=0.0))

    # the red pointed cuff, edged in gold along its top
    cuff = loft_region("Dolman_Cuff", A, lambda u, v: TAU * v, lambda u, v: u * cuff_top(TAU * v), closed=True, offset=cf["lift"],
                       mat=M["cuff"], coll=coll, res=kit.RES * 0.8,
                       attrs=dict(braid=lambda gr: line_value(cuff_top(gr.phi) - gr.t - cf["edge"])))
    mirror(solid(cuff, 0.003, bevel=0.0))

    # the standing collar, edged in gold
    c = g["collar"]
    lo = Loft([parts.ring(z, 0.0, c["cy"], c["a"] + 0.004 * (1 - k), c["bf"] + 0.004 * (1 - k), c["bb"] + 0.004 * (1 - k), 2.2, 1)
               for k, z in zip((0.0, 0.5, 1.0), np.linspace(*c["z"], 3))])
    col = loft_mesh("Dolman_Collar", lo, res=0.003, mat=M["dolman"], coll=coll,
                    attrs=dict(braid=lambda gr: line_value(np.minimum(np.abs(gr.t - 0.005), np.abs(gr.t - (lo.L - 0.005))))))
    solid(col, 0.004, bevel=0.0)

    # the gold shoulder cord on his RIGHT shoulder (his left is under the pelisse), collar ->
    # shoulder point, lying on the dolman's shoulder top
    sc = g["shoulder_cord"]
    trunk = workwear_trunk_field(F, 0.0075)
    xs = -np.linspace(sc[0][0] + 0.03, sc[1][0] + 0.01, 14)
    P0 = np.column_stack([xs, np.full(len(xs), 0.016), np.full(len(xs), 1.70)])
    braid.cord("Dolman_Shoulder_Cord", march(P0, -Z, trunk, step=0.02, iters=40) + Z * 0.004, r=0.0042, mat=M["braid"], coll=coll)
    # the pelisse's cord: from the right of the collar across the top of the chest to the
    # pelisse's front edge, a small gilt medallion on it (the painting)
    pc = g["pelisse_cord"]
    u = np.linspace(0.0, 1.0, 24)
    cx = pc["from"][0] + (pc["to"][0] - pc["from"][0]) * u
    cz = pc["from"][1] + (pc["to"][1] - pc["from"][1]) * u - pc["sag"] * np.sin(math.pi * u)
    Pcd, Ncd = half_body_pn(B, cx, cz, offset=0.011)
    braid.cord("Dolman_Pelisse_Cord", Pcd, r=0.0034, mat=M["braid"], coll=coll)
    k = int(pc["medal"] * (len(u) - 1))
    an = kit.OnPlane(Pcd[k] + Ncd[k] * 0.003, unit(Pcd[k + 1] - Pcd[k - 1]), np.cross(Ncd[k], unit(Pcd[k + 1] - Pcd[k - 1])))
    kit.plate("Dolman_Pelisse_Medallion", an, 0.010, 0.010, n=2.0, offset=0.003, thick=0.002, dome=0.003, mat=M["metal"], coll=coll)

    # buttons: the centre column and one each side, on every row of frogging
    fr = g["frogs"]
    zs = np.linspace(fr["z0"], fr["z1"], fr["n"])
    xs, zz = [], []
    for x in fr["buttons"]:
        for sg in ((1,) if x == 0 else (1, -1)):
            xs += [sg * x] * len(zs)
            zz += list(zs)
    P, N = half_body_pn(B, xs, zz, offset=0.0055)
    studs("Dolman_Buttons", P, N, r=fr["button_r"], flat=0.55, mat=M["metal"], coll=coll)
    return B


def workwear_trunk_field(F, off):
    """Signed distance (m) to the trunk loft's level sections, less `off` (for laying a cord on
    the dolman's shoulder), the same measure as `PelisseSurface.trunk_sdf`."""
    S = PelisseSurface.__new__(PelisseSurface)
    S.F, S.pd = F, {}
    return lambda Q: S.trunk_sdf(np.column_stack([np.abs(Q[:, 0]), Q[:, 1:]])) - off


# ------------------------------------------------------------ cross-belt, sash
def build_belt_and_sash(F, M, coll="Uniform"):
    g, B, bz = F.d["garment"], F.body, F.bz
    bl = g["belt"]
    u = np.linspace(0.0, 1.0, 48)
    x = bl["start"][0] + (bl["end"][0] - bl["start"][0]) * u
    z = bl["start"][1] + (bl["end"][1] - bl["start"][1]) * u
    P, N = half_body_pn(B, x, z, offset=bl["lift"])
    strap("Cross_Belt", P, N, bl["width"], thick=0.003, mat=M["braid"], coll=coll)
    k = int(bl["badge"] * (len(u) - 1))
    an = kit.OnPlane(P[k] + N[k] * 0.001, unit(P[k + 1] - P[k - 1]), np.cross(N[k], unit(P[k + 1] - P[k - 1])))
    kit.plate("Cross_Belt_Badge", an, 0.016, 0.012, n=2.4, offset=0.003, thick=0.002, dome=0.002, mat=M["metal"], coll=coll)

    sh = g["sash"]
    t0, t1 = bz(sh["z"][0]), bz(sh["z"][1])
    n_b = sh["barrels"]
    ph_b = np.linspace(D(-90), D(90), n_b + 2)[1:-1]

    def barrels(gr):
        dd = np.min(np.abs(wrap(gr.phi[..., None] - ph_b[None, None, :])), axis=-1) * gr.r
        return line_value(dd)

    sash = band_on("Sash", B, t0, t1, offset=sh["lift"], thick=sh["thick"], mat=M["sash"], coll=coll, res=0.003,
                   attrs=dict(barrel=barrels), **workwear.CLAMP_SNAP)
    mirror(sash, merge=True)
    pw, ph = sh["plate"]
    zc = 0.5 * (sh["z"][0] + sh["z"][1])
    Pp, Np = half_body_pn(B, [0.0], [zc], offset=sh["lift"] + 0.002)
    an = kit.OnPlane(Pp[0], np.array([1.0, 0.0, 0.0]), Z)
    kit.plate("Sash_Plate", an, 0.5 * pw, 0.5 * ph, n=6.0, offset=0.005, thick=0.004, dome=0.002, mat=M["metal"], coll=coll)

    # cords from the sash at his right hip, a ring, two tassels
    sc = g["sash_cords"]
    Ph, _ = half_body_pn(B, [sc["hang"][0]], [sc["hang"][1]], offset=sh["lift"] + 0.012)
    Pr, _ = half_body_pn(B, [sc["ring"][0]], [sc["ring"][1]], offset=0.022)
    braid.cord("Sash_Cord_Loop", [Ph[0], 0.5 * (Ph[0] + Pr[0]) + np.array([-0.008, -0.010, -0.005]), Pr[0]], r=sc["r"], mat=M["braid"], coll=coll)
    ring_pts = [Pr[0] + 0.012 * np.array([0.0, math.cos(a), math.sin(a)]) for a in np.linspace(0, TAU, 25)]
    kit.tube("Sash_Cord_Ring", np.array(ring_pts[:-1]), r=0.0028, closed=True, n_u=8, mat=M["braid"], coll=coll)
    for i, (x, z) in enumerate(sc["tassels"]):
        Pt, _ = half_body_pn(B, [x], [z + 0.002], offset=0.012 + 0.004 * i)
        braid.cord("Sash_Cord_%d" % i, [Pr[0], 0.5 * (Pr[0] + Pt[0]) + np.array([0.0, -0.006, 0.0]), Pt[0]], r=sc["r"], mat=M["braid"], coll=coll)
        tt = sc["tassel"]
        braid.tassel("Sash_Tassel_%d" % i, Pt[0], tt["length"], tt["r_head"], tt["r_skirt"], mat=M["braid"], coll=coll)


# ---------------------------------------------------------------------- pelisse
# the fur's pile: short, soft, lying along the roll (`fur.pile_field`), no tufts
FUR_PILE = dict(amp=0.0045, across=(0.008, 0.016), along=(0.020, 0.055), lean=0.5, waves=40, spike=1.5)


def _smin(a, b, k):
    """Smooth minimum (the union of two signed distances, filleted over `k`)."""
    return -kit.smax(-a, -b, k)


class PelisseSurface:
    """The slung pelisse's surface, derived from the surfaces it lies on.

    A carrier loft round the trunk and the left arm (`DIMS["pelisse"]["rings"]`) gives the
    parametrisation: (u, v) over the unit square, u from the hem up to the neck, v round from
    the edge across the back (0) by his left side to the front edge (1). Each carrier point is
    carried in along the carrier's normal onto the zero set of
        smin(sdf_trunk - clear_body, sdf_sleeve - clear_arm, blend)
    -- the dolman's trunk and left sleeve, filleted over the shoulder and bridged below the
    armpit where the pelisse hangs across from the side to the arm. So it lies ON the shoulder
    and arm by construction, never beside them."""

    def __init__(self, F, pd):
        self.F, self.pd = F, pd
        lo = self.lo = Loft([dict(p=(cx, cy, z), a=a, bf=bf, bb=bb, n=pd.get("n", 2.4), t=(0.0, 0.0, 1.0)) for z, cx, cy, a, bf, bb in pd["rings"]])
        L = self.L = lo.L
        ts = np.linspace(0.0, L, 240)
        zc = lo.frames(ts)[0][:, 2]
        self.tz = lambda z: np.interp(z, zc, ts)

        def edge_phi(t, x, front):
            ph = np.linspace(0.0, math.pi, 721) if front else np.linspace(math.pi, TAU, 721)
            X = lo.pos(ph, np.full(len(ph), t))[:, 0]
            o = np.argsort(X)
            return float(np.interp(x, X[o], ph[o]))

        fz, fx = zip(*pd["front_x"])
        bz_, bx = zip(*pd["back_x"])
        phi_f = np.array([edge_phi(t, np.interp(z, fz, fx), True) for t, z in zip(ts, zc)])
        phi_b = np.array([edge_phi(t, np.interp(z, bz_, bx), False) for t, z in zip(ts, zc)])
        self.pf = lambda t: np.interp(t, ts, phi_f)
        self.pb = lambda t: np.interp(t, ts, phi_b)
        # the hem: by the carrier point's world x at mid-height, on the back or the front half
        t_mid = self.tz(1.12)
        vv = np.linspace(0.0, 1.0, 400)
        ph = self.pb(t_mid) + vv * (self.pf(t_mid) + TAU - self.pb(t_mid))
        X = lo.pos(ph, np.full(len(ph), t_mid))[:, 0]
        hbx, hbz = zip(*pd["hem_back"])
        hfx, hfz = zip(*pd["hem_front"])
        zb, zf = np.interp(X, hbx, hbz), np.interp(X, hfx, hfz)
        w = np.clip(0.5 + 0.5 * np.sin(ph) / 0.35, 0.0, 1.0)  # 0 on the back, 1 on the front, blended round the side
        self._hem_t = self.tz(zb + (zf - zb) * w)
        self._vv = vv

    def t_lo(self, v):
        return np.interp(v, self._vv, self._hem_t)

    def t_of(self, u, v):
        return self.t_lo(v) + u * (self.L - self.t_lo(v))

    def phi_of(self, u, v):
        t = self.t_of(u, v)
        return self.pb(t) + v * (self.pf(t) + TAU - self.pb(t))

    # the field it lies on
    def trunk_sdf(self, Q):
        """Signed distance to the trunk loft, radially in its LEVEL section at each point's
        height (the trunk's rings are level above the crotch); above the neck ring the vertical
        distance joins in, so a point over the top is outside, not undefined."""
        B = self.F.body
        if not hasattr(self, "_tz_body"):
            ts = np.linspace(0.0, B.L, 1500)
            self._tz_body = (B.frames(ts)[0][:, 2], ts)
        zs, ts = self._tz_body
        z = Q[:, 2]
        t = np.interp(z, zs, ts)
        c, _, e1, e2, par = B.frames(t)
        # below the chest the cloth hangs plumb from it: the front and back depths do not
        # recede towards the waist under the pelisse
        zh = self.pd.get("hang_z")
        if zh is not None:
            below = z < zh
            if below.any():
                par_h = B.frames(np.full(int(below.sum()), np.interp(zh, zs, ts)))[4]
                par[below, 1:3] = np.maximum(par[below, 1:3], par_h[:, 1:3])
        D = Q - c
        x1, x2 = np.einsum("nk,nk->n", D, e1), np.einsum("nk,nk->n", D, e2)
        rad = np.hypot(x1, x2) - Loft._polar(np.arctan2(x2, x1), par)[0]
        over = np.maximum(z - zs[-1], 0.0)
        return np.where(over > 0, np.hypot(np.maximum(rad, 0.0), over), rad)

    def arm_sdf(self, Q):
        out = np.ones(len(Q))
        near = (Q[:, 0] > 0.10) & (Q[:, 2] > 0.85)
        if near.any():
            out[near] = kit.loft_sdf(self.F.arm, Q[near], m=300)
        return out

    def field(self, Q):
        c, bl = self.pd["clear"], self.pd["blend"]
        z = Q[:, 2]
        cback = c.get("back", c["body"])
        if "back_top" in c:  # the upper back, under the diagonal fur, lies closest; lower, it hangs
            cback = cback + (c["back_top"] - cback) * np.clip((z - 1.30) / 0.14, 0.0, 1.0)
        cb = c["body"] + (cback - c["body"]) * np.clip((Q[:, 1] - 0.02) / 0.06, 0.0, 1.0)
        cb = cb + (c["hem"] - cb) * np.clip((c["hem_z"] - z) / 0.06, 0.0, 1.0)
        if "top" in c:
            z0, z1 = c["top_z"]
            cb = cb + (c["top"] - cb) * np.clip((z - z0) / (z1 - z0), 0.0, 1.0)
        k = bl["top"] + (bl["low"] - bl["top"]) * np.clip((bl["z"][0] - z) / (bl["z"][0] - bl["z"][1]), 0.0, 1.0)
        return _smin(self.trunk_sdf(Q) - cb, self.arm_sdf(Q) - c["arm"], k)

    def carry(self, P, N, iters=22):
        """Distance s along -N from each carrier point to the pelisse's surface: damped sphere
        tracing with a step limit (the distances are radial, not Euclidean, so a ray that runs
        steeply down onto the shoulder sees them overstated)."""
        P, N = P.reshape(-1, 3), N.reshape(-1, 3)
        s = np.zeros(len(P))
        for i in range(iters):
            f = self.field(P - s[:, None] * N)
            s = np.clip(s + np.clip(0.75 * f, -0.012, 0.012), -0.03, 0.22)
        return s

    def at(self, U, V):
        """Points on the pelisse's surface at (u, v) arrays (flat), and their outward normals."""
        U, V = np.ravel(U), np.ravel(V)

        def pts(u, v):
            Pc, Nc = self.lo.pn(self.phi_of(u, v), self.t_of(u, v))
            return Pc - self.carry(Pc, Nc)[:, None] * Nc, Nc

        P, Nc = pts(U, V)
        h = 2e-3
        du = pts(np.clip(U + h, 0, 1), V)[0] - pts(np.clip(U - h, 0, 1), V)[0]
        dv = pts(U, np.clip(V + h, 0, 1))[0] - pts(U, np.clip(V - h, 0, 1))[0]
        N = unit(np.cross(du, dv))
        N = np.where((np.sum(N * Nc, axis=1) < 0)[:, None], -N, N)
        return P, N


def pelisse_lace(F, S):
    """The pelisse's gold lace: its own frogging across the front panel (rows from the front
    edge out over the shoulder, a loop at each end), and on its back panel the same piping as
    the dolman's back (two curved side-back seams, a centre seam, trefoils), so the back reads
    as one braided back on both sides of the diagonal fur."""
    pd, g = F.d["pelisse"], F.d["garment"]
    fr = pd["frogs"]
    fz, fx = zip(*pd["front_x"])
    rows = []
    for z in np.linspace(fr["z0"], fr["z1"], fr["n"]):
        x0 = float(np.interp(z, fz, fx)) + 0.020
        rows.append(np.array([[x0, z], [fr["x1"], z]]))
        rows.append(braid.loop_end(x0, z, -1, fr["loop"] * 0.8))
        rows.append(braid.loop_end(fr["x1"], z, 1, fr["loop"]))
    bk = g["back"]
    curve = smooth_path([[x, 0.0, z] for x, z in bk["curve"]], 30)[:, [0, 2]]
    back = [curve, np.array([[0.0, bk["centre"][0]], [0.0, bk["centre"][1]]])]
    for x, z in (bk["curve"][0], bk["curve"][-1]):
        back += braid.trefoil((x, z), bk["knot"], up=1.0 if z > 1.3 else -1.0)
    back += braid.trefoil((0.0, bk["centre"][1]), bk["knot"], up=-1.0)

    def f(gr):
        P, N = gr.Pp.reshape(-1, 3), gr.Np.reshape(-1, 3)
        d = np.ones(len(P))
        fr_ = N[:, 1] < -0.25
        if fr_.any():
            d[fr_] = polyline_dist(np.column_stack([P[fr_, 0], P[fr_, 2]]), rows)
        bk_ = N[:, 1] > 0.25
        if bk_.any():
            d[bk_] = polyline_dist(np.column_stack([np.abs(P[bk_, 0]), P[bk_, 2]]), back)
        return line_value(d.reshape(gr.Pp.shape[:-1]))

    return f


def build_pelisse(F, M, coll="Pelisse"):
    pd = F.d["pelisse"]
    S = PelisseSurface(F, pd)
    dr = pd["drape"]

    def disp(gr):
        sh = gr.P.shape
        P, N = gr.P.reshape(-1, 3), gr.N.reshape(-1, 3)
        s = S.carry(P, N)
        Pp = P - s[:, None] * N
        gr.Pp = Pp.reshape(sh)
        # normals of the carried surface, from the grid itself
        G = Pp.reshape(sh)
        du, dv = np.gradient(G, axis=0), np.gradient(G, axis=1)
        Np = unit(np.cross(du, dv))
        Np = np.where((np.sum(Np * gr.N, axis=-1) < 0)[..., None], -Np, Np)
        gr.Np = Np
        # drape: shallow vertical folds where it hangs free, near the hem, on the back and side
        z = G[..., 2]
        hem = S.pd["hem_back"][0][1]
        free = np.clip((dr["rise"] - (z - hem)) / dr["rise"], 0.0, 1.0) ** 1.5 * np.clip((Np[..., 1] + 0.2) / 0.4, 0.0, 1.0)
        folds = dr["amp"] * (0.5 + 0.5 * np.sin(gr.v * dr["n"] * TAU + 0.7)) * free
        return -s.reshape(sh[:-1]) + folds

    body = loft_region("Pelisse_Body", S.lo, S.phi_of, S.t_of, res=kit.RES * 0.9, disp=disp, attrs=dict(braid=pelisse_lace(F, S)),
                       mat=M["pelisse"], coll=coll)
    solid(body, pd["thick"], bevel=0.0)

    # fur: ONE roll round the whole edge -- the hem from his right hip across the back, round
    # the arm at elbow height to the front corner; up the front edge; round the back and left
    # of the neck (the collar, thick and round); down the diagonal across the back to the hip.
    # The corners are rounded (the control points near them are dropped).
    fu = pd["fur"]
    n = 64
    lin_ = np.linspace(0.0, 1.0, n)
    segs = [(lin_ * 0.0, lin_, "hem"),                    # u = 0, v 0 -> 1
            (lin_, np.ones(n), "edge"),                  # v = 1, u 0 -> 1 (the front edge)
            (np.ones(n), 1.0 - lin_, "collar"),          # u = 1, v 1 -> 0 (round the neck)
            (1.0 - lin_, np.zeros(n), "diag")]           # v = 0, u 1 -> 0 (the back diagonal)
    U = np.concatenate([s[0] for s in segs])
    V = np.concatenate([s[1] for s in segs])
    kind = sum([[s[2]] * n for s in segs], [])
    P, N = S.at(U, V)
    keep = np.ones(len(P), bool)
    corners = [k * n for k in range(1, 4)] + [k * n - 1 for k in range(1, 4)]
    for c in corners:  # round the corners: drop control points within a few cm of each
        keep &= np.linalg.norm(P - P[c], axis=1) > 0.035
    keep[0] = keep[-1] = True
    P, N = P[keep], N[keep]
    kind = [k for k, kp in zip(kind, keep) if kp]
    r0 = fu["hem"][0]
    rad = np.array([fu[k][0] for k in kind])
    flat = np.array([fu[k][1] for k in kind])
    # smooth the radius change along the roll
    rad = np.convolve(np.pad(rad, 6, mode="edge"), np.ones(13) / 13, mode="valid")
    flat = np.convolve(np.pad(flat, 6, mode="edge"), np.ones(13) / 13, mode="valid")
    C = P + N * (rad * flat * 0.62)[:, None]
    # the collar stretch: a round roll hugging the neck (its centre on a ring round the neck's
    # axis), blended into the edge rolls where they leave it
    col = np.array([k == "collar" for k in kind], float)
    col = np.convolve(np.pad(col, 5, mode="edge"), np.ones(11) / 11, mode="valid")
    axis = np.array([0.0, F.d["garment"]["collar"]["cy"]])
    radial = unit(np.column_stack([C[:, 0] - axis[0], C[:, 1] - axis[1], np.zeros(len(C))]))
    Cn = np.column_stack([axis[0] + radial[:, 0] * fu["ring"], axis[1] + radial[:, 1] * fu["ring"], np.full(len(C), fu["z"])])
    C = C + col[:, None] * (Cn - C)
    N = unit(N + col[:, None] * (radial + np.array([0.0, 0.0, 0.6]) - N))
    T = np.gradient(C, axis=0)
    across = unit(np.cross(N, T))
    fur.fur_roll("Pelisse_Fur", C, r0, seed=11, flat=flat, up=across, scale=rad / r0, soft=FUR_PILE, mat=M["fur"], coll=coll)

    # its frogging's buttons: one at the inner loop of each row, on the carried surface
    fr = pd["frogs"]
    fz, fx = zip(*pd["front_x"])
    zs = np.linspace(fr["z0"], fr["z1"], fr["n"])
    P0 = np.column_stack([np.interp(zs, fz, fx) + 0.020 + 0.008, np.full(len(zs), -0.30), zs])
    Pb = march(P0, np.array([0.0, 1.0, 0.0]), S.field)
    studs("Pelisse_Buttons", Pb + np.array([0.0, -0.002, 0.0]), np.tile([0.0, -1.0, 0.0], (len(zs), 1)), r=0.0050, flat=0.7, mat=M["metal"], coll=coll)
    # the gold shoulder cord on its shoulder, from the collar to the shoulder point
    sc = F.d["garment"]["shoulder_cord"]
    xs = np.linspace(sc[0][0] + 0.03, sc[1][0] + 0.03, 14)
    P0 = np.column_stack([xs, np.full(len(xs), 0.016), np.full(len(xs), 1.70)])
    Pc = march(P0, np.array([0.0, 0.0, -1.0]), S.field, step=0.02, iters=40)
    braid.cord("Pelisse_Shoulder_Cord", Pc + np.array([0.0, 0.0, 0.004]), r=0.0042, mat=M["braid"], coll=coll)

    # the empty sleeve (`pd["sleeves"]`): a flattened tube along its centre line, a fur cuff
    # over its last `cuff` metres, gold chevrons on its broad face towards `face`
    for k, sl in enumerate(pd["sleeves"]):
        build_empty_sleeve("Pelisse_Sleeve_%s" % sl["name"], sl, M, coll, seed=15 + k)
    return S


def march(P0, d, field, step=0.012, iters=30, off=0.0):
    """Points carried from P0 along the direction d onto the zero set of `field` (+ `off`)."""
    P0 = np.asarray(P0, float)
    s = np.zeros(len(P0))
    for _ in range(iters):
        s = s + np.clip(0.8 * (field(P0 + s[:, None] * d) - off), -step, step)
    return P0 + s[:, None] * d


def build_empty_sleeve(name, sl, M, coll, seed=15):
    c = smooth_path(sl["path"], 60)
    Ls = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
    flat = sl["b"] / sl["a"]
    wide = unit(np.asarray(sl["wide"], float))
    T = unit(np.gradient(c, axis=0))
    Nn = unit(wide[None, :] - (T @ wide)[:, None] * T)
    Bn = np.cross(T, Nn)
    al_c = 0.5 * math.pi * (1.0 if Bn[len(c) // 2] @ np.asarray(sl["face"], float) >= 0 else -1.0)
    t_c = Ls - sl["cuff"]
    chev = sum([bold_v(0.0, t_c - 0.014 - 0.030 * i, 0.048, 0.034, 3, 0.004) for i in range(2)], [])
    r_m = 0.5 * (sl["a"] + sl["b"])

    def lace(UU, AL):  # chevrons point up the sleeve, drawn on the broad face round angle al_c
        Q = np.column_stack([(wrap(AL - al_c) * r_m).ravel(), (UU * Ls).ravel()])
        return line_value(polyline_dist(Q, chev).reshape(UU.shape))

    # limp: an EMPTY sleeve -- flattened, wider at the top than at the cuff, its two faces
    # lying in a few soft lengthwise creases, the section a little uneven along it
    uu = np.linspace(0.0, 1.0, len(c))
    radii = sl["a"] * (1.08 - 0.16 * uu)
    flats = flat * (1.0 + 0.25 * np.sin(uu * 7.0 + seed))
    creases = lambda u, al: (0.0035 * np.sin(2.0 * al) ** 2 * np.sin(3.0 * al + 1.3 * seed + 2.0 * u)
                             - 0.0025 * np.exp(-((wrap(al - al_c) - 0.4) / 0.25) ** 2) * np.clip(u / 0.2, 0, 1))
    swept(name, c, radii, n_u=48, flat=flats, up=Nn, disp=creases, attrs=dict(braid=lace), mat=M["pelisse"], coll=coll)
    # the fur cuff: a soft roll round the sleeve's end (a ring following its flattened
    # section), and a second one a little higher, so it reads as a deep fur cuff, not a drum
    for j, back in enumerate((0.022, 0.052)):
        k = int(np.clip((1.0 - back / Ls) * (len(c) - 1), 0, len(c) - 1))
        al = np.linspace(0.0, TAU, 41)
        rr = radii[k] + 0.004
        ring = c[k] + rr * (np.cos(al)[:, None] * Nn[k] + (flats[k] * np.sin(al))[:, None] * Bn[k])
        ring_up = unit(c[k] - ring)  # across the roll: towards the sleeve's centre line
        fur.fur_roll(name + "_Fur_%d" % j, ring, 0.020, seed=seed + j, flat=0.75, up=ring_up, soft=FUR_PILE, mat=M["fur"], coll=coll)


def _outward(lo, phi, t):
    """Per point: the direction ALONG the surface across a fur roll lying on it, so its
    `flat` squashes it towards the surface (the roll's frame: first axis = this)."""
    P, N = lo.pn(phi, t)
    T = np.gradient(P, axis=0)
    return unit(np.cross(N, T))


# ------------------------------------------------------------------- hands, boots
def build_hands(F, M, coll="Hands"):
    h = F.d["hand"]
    w = np.array(F.d["sleeve"][0][1:4], float)
    L = unit(h["down"])
    b = np.asarray(h["back"], float)
    Bv = unit(b - (b @ L) * L)
    parts.hand("Glove", w + L * h["drop"], L, Bv, M["glove"], coll, scale=h["scale"], curl=h["curl"], girth=h["girth"], palm_girth=h["palm_girth"])
    garment.close_holes([o for o in bpy.data.objects if o.name.startswith("Glove_")])


def build_boots(F, M, coll="Boots"):
    s = F.d["boot"]
    bt, objs = dresswear.riding_boot("Boot", s, dict(leather=M["boot"], sole=M["sole"], trim=M["braid"]), coll=coll)
    t_notch = float(bt.top_t(np.array([D(90.0)]))[0])
    P, N = bt.shaft.pn([D(90.0)], [t_notch], 0.006)
    ts = s["tassel"]
    mirror(braid.tassel("Boot_Tassel", P[0] - Z * 0.004, ts["length"], ts["r_head"], ts["r_skirt"], mat=M["braid"], coll=coll))
    dresswear.spur("Boot", bt, s["spur"], M["steel"], coll=coll)
    return bt


# ------------------------------------------------------------------------- head
def build_head(M, head, hair, coll="Head"):
    loft, _ = parts.head("Guard", head, dict(skin=M["skin"], eye=M["eye"]), coll=coll)
    parts.hair("Guard", loft, hair, M["hair"], coll=coll)
    ms = head["moustache"]
    rows = ms["rows"]
    xs = [-r[0] for r in rows[::-1]] + [r[0] for r in rows[1:]]
    dzs = [r[1] for r in rows[::-1]] + [r[1] for r in rows[1:]]
    rad = [r[2] for r in rows[::-1]] + [r[2] for r in rows[1:]]
    pts, nrm = [], []
    for x, dz, rr in zip(xs, dzs, rad):
        t = loft.t_at_z(head["mouth_z"] + dz)
        ph = parts.phi_at_x(loft, t, x)
        P, N = loft.pn([ph], [t], ms["lift"] + 0.6 * rr + _face_height(head, x, head["mouth_z"] + dz))
        pts.append(P[0])
        nrm.append(N[0])
    pts, nrm = np.array(pts), np.array(nrm)
    n = 40
    tk = np.linspace(0.0, 1.0, len(pts))
    q = np.linspace(0.0, 1.0, n)
    c = kit.spline(tk, pts, q)
    rr = np.interp(q, tk, rad)
    nn = unit(kit.spline(tk, nrm, q))
    T = unit(np.gradient(c, axis=0))
    up = unit(np.cross(nn, T))
    swept("Guard_Moustache", c, rr, n_u=14, flat=ms["flat"], up=up, mat=M["hair"], coll=coll,
          disp=lambda u, al: 0.0006 * np.sin(al * 7.0 + u * 40.0), attrs=dict(cover=lambda u, al: np.ones(u.shape)))
    return loft


def _face_height(head, x, z):
    """The face field's displacement at the middle of the upper lip (lips and philtrum), so
    the moustache sits ON the lip, not inside it: the same rows `parts.FACE` uses."""
    d = 0.0
    for key in ("lip_upper", "philtrum", "mouth"):
        for x0, (lm, dz), sx, sz, h in parts.FACE[key]:
            zc = dict(mouth=head["mouth_z"], nose=head["nose_z"], eye=head["eye_z"], chin=head["chin_z"])[lm] + dz
            d += h * math.exp(-((abs(x) - x0) / sx) ** 2 - ((z - zc) / sz) ** 2)
    return max(d, 0.0)


def build_shako(M, spec, head_loft, plumes, coll="Shako"):
    lo, _ = shako.shako("Shako", spec, dict(body=M["shako"], gold=M["braid"], metal=M["metal"], peak=M["peak"], cockade=M["cockade"]), coll=coll)
    shako.chin_chain("Shako", lo, head_loft, spec, M["metal"], coll=coll)
    for i, pl in enumerate(plumes):
        fur.plume_soft("Plume_%d" % i, pl, seed=21 + i, mat=M["plume_%d" % i], coll="Plumes")
    return lo


# ----------------------------------------------------------------------- figure
def build(M, dims, head, hair, plumes):
    F = workwear.Figure(dims)
    build_body(F, M)
    build_belt_and_sash(F, M)
    build_hands(F, M)
    build_boots(F, M)
    hl = build_head(M, head, hair)
    build_shako(M, dims["shako"], hl, plumes)
    build_pelisse(F, M)
    garment.close_holes([o for o in bpy.data.objects if o.name.startswith(("Dolman_Buttons",))])
    return F
