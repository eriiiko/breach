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
        dolman=dressmat.mat_wool("guard_dolman", p["DOLMAN"], lace=p["GOLD"]),
        pelisse=dressmat.mat_wool("guard_pelisse", p["PELISSE"], lace=p["GOLD"]),
        cuff=dressmat.mat_wool("guard_cuff", p["CUFF"], lace=p["GOLD"]),
        breeches=dressmat.mat_wool("guard_breeches", p["BREECHES"], lace=p["GOLD"]),
        fur=dressmat.mat_fur("guard_fur", p["FUR"], p["FUR_TIP"]),
        braid=dressmat.mat_gold_cord("guard_braid", p["GOLD"]),
        metal=dressmat.mat_gold_metal("guard_metal", p["METAL"]),
        steel=wearmat.mat_metal("guard_steel", p["STEEL"], rough=0.30),
        sash=dressmat.mat_sash("guard_sash", p["SASH"], p["SASH_STRIPE"], p["GOLD"]),
        boot=dressmat.mat_polished("guard_boot", p["BOOT"]),
        sole=wearmat.mat_rubber("guard_sole", p["SOLE"], rough=0.6),
        glove=dressmat.mat_glove("guard_glove", p["GLOVE"]),
        shako=dressmat.mat_shako("guard_shako", p["SHAKO"]),
        peak=dressmat.mat_polished("guard_peak", p["PEAK"], rough=0.12),
        cockade=dressmat.mat_wool("guard_cockade", p["COCKADE"], rough=0.6),
    )
    for i, pl in enumerate(plumes):
        M["plume_%d" % i] = dressmat.mat_feather("guard_plume_" + "abcd"[i], p[pl["colour"]], p[pl["tip"]] if pl.get("tip") else None,
                                                 pl.get("tip_from", 0.75))
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


def breeches_lace(F, B):
    g = F.d["garment"]
    kn = g["knot"]
    knot = braid.hungarian_knot((kn["x"], kn["top"]), kn["height"], kn["width"])
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
    ff_b = fold_field(fold_set(rng, 6, (bz(kz - 0.04), bz(kz + 0.06)), (D(215), D(325)), (0.0, 0.25), (0.03, 0.06), (0.004, 0.007), (0.0015, 0.003))
                      + fold_set(rng, 5, (bz(0.75), bz(0.84)), (D(40), D(140)), (-0.4, 0.3), (0.04, 0.08), (0.006, 0.010), (0.0015, 0.003))
                      + crumple_set(rng, 160, (0.0, bz(z_btop))))
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
    cuff_top = lambda phi: t_cuff + cf["point"] * np.clip(1.0 - np.abs(wrap(phi)) / D(cf["point_w"]), 0.0, 1.0)
    ch = g["chevrons"]
    r_c = float(A.radius([0.0], [t_cuff])[0])
    chev = []
    for i in range(ch["n"]):
        tp = t_cuff + cf["point"] + (i + 1) * ch["gap"]
        chev.append(np.array([[-ch["half"], tp - ch["drop"]], [0.0, tp], [ch["half"], tp - ch["drop"]]]))

    def sleeve_lace(gr):
        Q = np.column_stack([(wrap(gr.phi) * gr.r).ravel(), gr.t.ravel()])
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
def build_pelisse(F, M, coll="Pelisse"):
    pd = F.d["pelisse"]
    lo = Loft([dict(p=(cx, cy, z), a=a, bf=bf, bb=bb, n=pd.get("n", 2.4), t=(0.0, 0.0, 1.0)) for z, cx, cy, a, bf, bb in pd["rings"]])
    L = lo.L
    ts = np.linspace(0.0, L, 240)
    zc = lo.frames(ts)[0][:, 2]
    tz = lambda z: np.interp(z, zc, ts)
    bx_z, bx_x = zip(*pd["back_x"])

    def edge_phi(t, x, front):
        ph = np.linspace(0.0, math.pi, 721) if front else np.linspace(math.pi, TAU, 721)
        X = lo.pos(ph, np.full(len(ph), t))[:, 0]
        o = np.argsort(X)
        return float(np.interp(x, X[o], ph[o]))

    phi_f = np.array([edge_phi(t, pd["front_x"], True) for t in ts])
    phi_b = np.array([edge_phi(t, np.interp(z, bx_z, bx_x), False) for t, z in zip(ts, zc)])
    pf = lambda t: np.interp(t, ts, phi_f)
    pb = lambda t: np.interp(t, ts, phi_b)
    hb, hf = pd["hem"]
    t_mid = tz(0.5 * (hb + hf))

    def t_lo(v):
        ph = pb(t_mid) + v * (pf(t_mid) + TAU - pb(t_mid))
        return tz(hb + (hf - hb) * np.clip(np.sin(ph), 0.0, 1.0))

    t_of = lambda u, v: t_lo(v) + u * (L - t_lo(v))

    def phi_of(u, v):
        t = t_of(u, v)
        return pb(t) + v * (pf(t) + TAU - pb(t))

    fr = pd["frogs"]
    x0 = pd["front_x"] + 0.016
    rows = []
    for z in np.linspace(fr["z0"], fr["z1"], fr["n"]):
        rows.append(np.array([[x0, z], [x0 + fr["w"], z]]))
        rows.append(braid.loop_end(x0, z, -1, fr["loop"] * 0.8))
        rows.append(braid.loop_end(x0 + fr["w"], z, 1, fr["loop"]))

    def lace(gr):
        P = gr.P.reshape(-1, 3)
        Q = np.column_stack([P[:, 0], P[:, 2]])
        dd = np.where(P[:, 1] < -0.03, polyline_dist(Q, rows), 1.0)
        return line_value(dd.reshape(gr.P.shape[:-1]))

    def drape(gr):
        z = gr.P[..., 2]
        return 0.004 * np.sin(gr.phi * 11.0 + 0.7) * np.clip((1.32 - z) / 0.2, 0.0, 1.0)

    body = loft_region("Pelisse_Body", lo, phi_of, t_of, res=kit.RES * 0.9, disp=drape, attrs=dict(braid=lace), mat=M["pelisse"], coll=coll)
    solid(body, 0.003, bevel=0.0)

    # fur: the front edge, the hem, the edge across the back, the collar round the neck
    fu = pd["fur"]
    r = fu["r"]
    tt = np.linspace(t_lo(1.0), L, 30)
    P, _ = lo.pn(pf(tt), tt, 0.4 * r)
    fur.fur_roll("Pelisse_Fur_Front", P, r, seed=11, flat=fu["flat"], up=_outward(lo, pf(tt), tt), mat=M["fur"], coll=coll)
    tt = np.linspace(t_lo(0.0), L, 30)
    P, _ = lo.pn(pb(tt), tt, 0.4 * r)
    fur.fur_roll("Pelisse_Fur_Back", P, r, seed=12, flat=fu["flat"], up=_outward(lo, pb(tt), tt), mat=M["fur"], coll=coll)
    vv = np.linspace(0.0, 1.0, 60)
    th = t_lo(vv)
    ph_h = pb(th) + vv * (pf(th) + TAU - pb(th))
    P, _ = lo.pn(ph_h, th, 0.4 * r)
    fur.fur_roll("Pelisse_Fur_Hem", P, r, seed=13, flat=fu["flat"], up=_outward(lo, ph_h, th), mat=M["fur"], coll=coll)
    # the shawl collar: a band of fur lying over the top of the pelisse, round the neck
    t_col = tz(fu["collar_z"])
    tcol_of = lambda u, v: t_col + u * (L - t_col)
    fur.fur_patch("Pelisse_Fur_Collar", lo, lambda u, v: pb(tcol_of(u, v)) - 0.05 + v * (pf(tcol_of(u, v)) + TAU + 0.10 - pb(tcol_of(u, v))), tcol_of,
                  fu["collar_h"], seed=14, mat=M["fur"], coll=coll)

    # the two empty sleeves (`pd["sleeves"]`): a flattened tube along each centre line, a fur
    # cuff over its last `cuff` metres, gold chevrons on its broad face towards `face`
    for k, sl in enumerate(pd["sleeves"]):
        build_empty_sleeve("Pelisse_Sleeve_%s" % sl["name"], sl, M, coll, seed=15 + k)
    return lo


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
    chev = [np.array([[-0.045, t_c - 0.012 - 0.024 * i - 0.030], [0.0, t_c - 0.012 - 0.024 * i], [0.045, t_c - 0.012 - 0.024 * i - 0.030]])
            for i in range(2)]
    r_m = 0.5 * (sl["a"] + sl["b"])

    def lace(UU, AL):  # chevrons point up the sleeve, drawn on the broad face round angle al_c
        Q = np.column_stack([(wrap(AL - al_c) * r_m).ravel(), (UU * Ls).ravel()])
        return line_value(polyline_dist(Q, chev).reshape(UU.shape))

    swept(name, c, sl["a"], n_u=48, flat=flat, up=Nn, attrs=dict(braid=lace), mat=M["pelisse"], coll=coll)
    k0 = int((1.0 - sl["cuff"] / Ls) * (len(c) - 1))
    fur.fur_roll(name + "_Fur", c[k0:], sl["a"] + 0.012, seed=seed, flat=(sl["b"] + 0.012) / (sl["a"] + 0.012), up=Nn[k0:], mat=M["fur"], coll=coll)


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
        fur.plume("Plume_%d" % i, pl, seed=21 + i, mat=M["plume_%d" % i], coll="Plumes")
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
