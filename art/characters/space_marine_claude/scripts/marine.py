"""The space marine, after `art/characters/Space Marine Turnaround Sheet.png`.

Proportions are fitted to the sheet's measured silhouette (`ref_measure.py`):
1.88 m to the helmet top, character faces -Y, +X is the character's left.
Limbs are modelled on the LEFT and mirrored. On a left limb `phi` reads
0 = outer side, 90 deg = front, 180 deg = inner side, 270 deg = back.
"""
import math

import numpy as np
from mathutils import Matrix, Vector

import kit
from kit import (TAU, Loft, OnEllipsoid, OnLoft, OnPlane, Oval, band_mask, band_on, box, box_at, crumple_set, fold_field, fold_set,
                 loft_mesh, mirror, plate, seam_attr, solid, studs, tube, unit, wrap)

D = math.radians
OUT, FRONT, IN, BACK = 0.0, D(90), D(180), D(270)
Z = np.array([0.0, 0.0, 1.0])


def vring(z, a, bf, bb, n=2.6, x=0.0, y=0.0, pin=False):
    r = dict(p=(x, y, z), a=a, bf=bf, bb=bb, n=n)
    if pin:
        r["t"] = (0.0, 0.0, 1.0)
    return r


def rivets(name, anchor, st, off, r=0.0035, mat=None, coll="Marine"):
    s, t = zip(*st)
    P, N = anchor.pn(s, t, off)
    return studs(name, P, N, r, mat=mat, coll=coll)


# ----------------------------------------------------------------- body surfaces
TORSO = Loft([vring(*r) for r in (
    (1.10, .200, .160, .150, 2.7), (1.17, .202, .165, .148, 2.7), (1.24, .200, .168, .145, 2.7),
    (1.31, .212, .172, .148, 2.7), (1.38, .226, .176, .152, 2.7), (1.45, .232, .170, .152, 2.6),
    (1.51, .215, .150, .146, 2.5), (1.56, .175, .128, .132, 2.4), (1.61, .135, .112, .118, 2.3),
    (1.67, .110, .100, .105, 2.2))])

# One trouser leg, ankle to waist. Above the crotch the section grows into half the
# pelvis and is clamped at the mid-plane, where its mirror image is welded on: the
# trousers are one surface with a real centre seam, not two tubes pushed into a hip.
LEG = Loft([vring(z, a, bf, bb, n, x, y, pin) for z, x, y, a, bf, bb, n, pin in (
    (0.20, .244, .046, .076, .094, .092, 2.0, 0), (0.26, .235, .050, .082, .094, .092, 2.0, 0),
    (0.32, .228, .050, .089, .097, .099, 2.0, 0), (0.40, .215, .050, .096, .100, .106, 2.0, 0),
    (0.48, .200, .032, .096, .104, .108, 2.0, 0), (0.56, .186, .012, .093, .102, .104, 2.0, 0),
    (0.64, .169, -.002, .100, .116, .116, 2.0, 0), (0.72, .152, -.010, .110, .126, .128, 2.0, 0),
    (0.80, .137, -.012, .113, .134, .136, 2.1, 0), (0.86, .124, -.010, .124, .142, .142, 2.2, 1),
    (0.90, .100, -.006, .148, .150, .150, 2.3, 1), (0.94, .070, -.003, .176, .154, .158, 2.4, 1),
    (1.02, .030, .000, .197, .158, .168, 2.5, 1), (1.10, .000, .000, .210, .162, .160, 2.6, 1),
    (1.15, .000, .000, .206, .164, .152, 2.6, 1))])

# Sleeve, wrist to shoulder, turning inwards over the shoulder like a raglan sleeve.
ARM = Loft([dict(p=p, a=a, b=b, n=2.0) for p, a, b in (
    ((.447, -.037, 0.926), .050, .054), ((.428, -.022, 1.000), .057, .060), ((.402, .010, 1.080), .071, .074),
    ((.376, .038, 1.150), .077, .080), ((.352, .055, 1.215), .079, .083), ((.328, .050, 1.280), .081, .086),
    ((.294, .034, 1.360), .086, .092), ((.264, .018, 1.425), .090, .096), ((.234, .010, 1.466), .090, .096),
    ((.190, .008, 1.494), .082, .092), ((.135, .006, 1.508), .070, .085))])
T_ELBOW = ARM.t_ring(4)

BELT = Loft([vring(z, a, bf, bb, 2.7) for z, a, bf, bb in (
    (1.097, .200, .158, .148), (1.103, .214, .172, .160), (1.137, .215, .173, .160),
    (1.171, .214, .172, .160), (1.177, .200, .158, .148))])

# Collar mantle: slopes from the neck ring down onto the shoulder tops.
YOKE = Loft([vring(z, a, bf, bb, 2.6, y=-0.004) for z, a, bf, bb in (
    (1.534, .258, .146, .146), (1.544, .276, .160, .158), (1.568, .262, .163, .160),
    (1.594, .226, .158, .156), (1.610, .190, .150, .146), (1.615, .172, .142, .138))])

NECK_RING = Loft([vring(z, a, bf, bb, 2.5, y=-0.012) for z, a, bf, bb in (
    (1.600, .158, .150, .146), (1.607, .180, .160, .158), (1.640, .183, .162, .160),
    (1.660, .175, .160, .154), (1.667, .158, .156, .146))])

# Helmet outer shell: an ellipsoid dome whose lower third stays wide (jaw guard).
HC = np.array([0.0, -0.018, 1.733])
HA, HB, HCZ, HZ0 = 0.143, 0.156, 0.147, 1.632


def _helmet_fn(t):
    z = HZ0 + t
    u = np.clip((z - HC[2]) / HCZ, -1.0, 1.0)
    k = np.sqrt(np.maximum(1.0 - np.where(u < 0, 0.75 * u, u) ** 2, 4e-5))
    n = len(t)
    c = np.column_stack([np.zeros(n), np.full(n, HC[1]), z])
    par = np.column_stack([HA * k, HB * k, HB * k, np.full(n, 2.15), np.full(n, 2.15)])
    return c, np.tile(Z, (n, 1)), par


HELMET = Loft(fn=_helmet_fn, length=HC[2] + HCZ - HZ0)


def helmet_rows(res):
    b0 = math.asin((HZ0 - HC[2]) / HCZ)
    beta = np.linspace(b0, math.pi / 2 - 0.012, int(round((math.pi / 2 - b0) * HCZ / res)))
    return HC[2] + HCZ * np.sin(beta) - HZ0


# ------------------------------------------------------------------------- suit
def build_suit(M):
    C = "Suit"
    # torso ------------------------------------------------------------------
    t = TORSO.t_at_z
    rng = np.random.default_rng(11)
    ff = fold_field(fold_set(rng, 18, (t(1.19), t(1.31)), (0, TAU), (0.0, 0.16), (0.05, 0.12), (0.006, 0.010), (0.003, 0.006))
                    + fold_set(rng, 9, (t(1.32), t(1.50)), (D(200), D(340)), (0.0, 0.5), (0.04, 0.09), (0.006, 0.010), (0.002, 0.005))
                    + crumple_set(rng, 240, (0.0, t(1.56))))

    def flex(g):
        side = np.minimum(np.abs(wrap(g.phi - OUT)), np.abs(wrap(g.phi - IN))) * g.r
        z = g.P[..., 2]
        return band_mask(g, np.maximum.reduce([side - 0.062, 1.168 - z, z - 1.43]))

    loft_mesh("Torso", TORSO, coll=C, mat=M["suit"],
              disp=lambda g: ff(g) * (1 - flex(g)) - 0.003 * flex(g),
              attrs=dict(flex=flex, seam=lambda g: seam_attr(
                  g, phis=[(FRONT, 0, t(1.31)), (FRONT - D(26), 0, t(1.31)), (FRONT + D(26), 0, t(1.31)), BACK,
                           (BACK - D(38), 0, t(1.32)), (BACK + D(38), 0, t(1.32))],
                  ts=[(t(1.305), FRONT, D(50))])))

    # trousers ---------------------------------------------------------------
    t = LEG.t_at_z
    rng = np.random.default_rng(23)
    ff_leg = fold_field(
        fold_set(rng, 13, (t(0.225), t(0.43)), (0, TAU), (0.0, 0.14), (0.06, 0.15), (0.007, 0.012), (0.005, 0.009))  # stacking on the boot
        + fold_set(rng, 6, (t(0.48), t(0.66)), (D(215), D(325)), (0.0, 0.2), (0.05, 0.09), (0.006, 0.009), (0.003, 0.005))  # knee hollow
        + fold_set(rng, 7, (t(0.67), t(0.84)), (D(40), D(165)), (-0.45, 0.3), (0.05, 0.10), (0.007, 0.011), (0.003, 0.006))  # thigh
        + fold_set(rng, 4, (t(0.87), t(0.97)), (D(60), D(125)), (-0.55, 0.12), (0.07, 0.12), (0.008, 0.012), (0.004, 0.007))  # groin
        + fold_set(rng, 5, (t(0.80), t(0.90)), (D(230), D(300)), (0.15, 0.2), (0.06, 0.10), (0.008, 0.012), (0.004, 0.007))  # under the seat
        + fold_set(rng, 5, (t(0.44), t(0.50)), (D(30), D(150)), (0.0, 0.2), (0.04, 0.07), (0.006, 0.009), (0.003, 0.005))
        + crumple_set(rng, 340, (0.0, LEG.L)))

    def flex_leg(g):
        z = g.P[..., 2]
        knee = np.maximum(0.468 - z, z - 0.662)
        inner = np.abs(wrap(g.phi - IN)) * g.r
        gusset = np.maximum.reduce([inner - (0.030 + 0.32 * (z - 0.70)), 0.70 - z, z - 0.935])
        return band_mask(g, np.minimum(knee, gusset))

    def seam_leg(g):
        z = g.P[..., 2]
        centre = np.where(z > 0.84, np.maximum(g.P[..., 0], 0.0), 1.0)
        return seam_attr(g, phis=[OUT, (IN, 0, t(0.70))], ts=[t(0.224)], extra=centre)

    legs = loft_mesh("Trousers", LEG, coll=C, mat=M["suit"],
                     disp=lambda g: ff_leg(g) * (1 - flex_leg(g)) - 0.003 * flex_leg(g),
                     attrs=dict(flex=flex_leg, seam=seam_leg),
                     drop=lambda P: P[:, 0] <= 1e-6,
                     post=lambda P: np.column_stack([np.maximum(P[:, 0], 0.0), P[:, 1:]]))
    mirror(legs, merge=True)
    mirror(band_on("Trouser_Hem", LEG, 0.0, 0.022, offset=0.004, thick=0.009, bevel=0.003, mat=M["suit"], coll=C))

    # sleeves ----------------------------------------------------------------
    rng = np.random.default_rng(37)
    ff_arm = fold_field(
        fold_set(rng, 7, (0.012, 0.12), (0, TAU), (0.0, 0.15), (0.04, 0.10), (0.006, 0.010), (0.004, 0.007))
        + fold_set(rng, 6, (T_ELBOW - 0.11, T_ELBOW - 0.05), (D(20), D(200)), (0.0, 0.25), (0.04, 0.08), (0.006, 0.009), (0.003, 0.006))
        + fold_set(rng, 8, (T_ELBOW + 0.06, ARM.t_ring(7)), (0, TAU), (0.2, 0.4), (0.04, 0.09), (0.006, 0.010), (0.003, 0.006))
        + crumple_set(rng, 170, (0.0, ARM.t_ring(8))))

    def flex_arm(g):
        elbow = np.maximum(T_ELBOW - 0.050 - g.t, g.t - (T_ELBOW + 0.055))
        top = ARM.t_ring(8) - 0.010 - g.t  # the shoulder root, under pauldron and mantle
        pit = np.maximum(ARM.t_ring(6) - g.t, np.abs(wrap(g.phi - IN)) * g.r - 0.11)  # armpit side
        return band_mask(g, np.minimum.reduce([elbow, top, pit]))

    mirror(loft_mesh("Sleeve", ARM, coll=C, mat=M["suit"],
                     disp=lambda g: ff_arm(g) * (1 - flex_arm(g)) - 0.003 * flex_arm(g),
                     attrs=dict(flex=flex_arm, seam=lambda g: seam_attr(g, phis=[(BACK, 0, ARM.t_ring(7)), (FRONT, 0, ARM.t_ring(7))]))))


# ------------------------------------------------------------------------ armour
def build_armor(M):
    C = "Armor"
    arm, suit, dark = M["armor"], M["suit"], M["dark"]
    t = TORSO.t_at_z

    # chest: padded bib, projecting breast plate, sockets and vent slot
    chest = lambda z: OnLoft(TORSO, FRONT, t(z))
    plate("Chest_Bib", chest(1.425), 0.205, 0.125, n=4.5, offset=0.013, thick=0.012, bevel=0.004, subsurf=1, mat=suit, coll=C)
    plate("Chest_Plate", chest(1.396), 0.146, 0.088, n=6.0, offset=0.028, thick=0.018, dome=0.004, bevel=0.005, seg=3, mat=arm, coll=C)
    plate("Chest_Slot", chest(1.396), 0.038, 0.005, n=4.0, shift=(0, -0.060), offset=0.0315, thick=0.004, mat=dark, coll=C)
    for sx in (-0.100, 0.100):
        plate("Chest_Socket", chest(1.396), 0.017, 0.011, n=4.0, shift=(sx, 0.058), offset=0.0315, thick=0.004, mat=dark, coll=C)
        box_at("Chest_Buckle", chest(1.43), (0.022, 0.042, 0.014), shift=(sx * 1.74, 0.0), sink=-0.011, bevel=0.003, mat=dark, coll=C)
    rivets("Chest_Rivets", chest(1.396), [(sx, sy) for sx in (-0.128, 0.128) for sy in (-0.070, 0.070)], 0.029, mat=M["metal"], coll=C)
    band_on("Chest_Strap", TORSO, t(1.412), t(1.448), offset=0.006, thick=0.004, mat=suit, coll=C)

    # shoulder: curved plate with the recessed steel-blue field
    sh = OnEllipsoid((0.234, 0.010, 1.458), (0.100, 0.106, 0.106), n0=(0.82, -0.02, 0.57))
    mirror(plate("Pauldron", sh, 0.092, 0.112, n=4.0, offset=0.014, thick=0.012, hole=0.70, bevel=0.003, seg=3, mat=arm, coll=C))
    mirror(plate("Pauldron_Field", sh, 0.092 * 0.74, 0.112 * 0.74, n=4.0, offset=0.0095, thick=0.008, mat=M["blue"], coll=C))
    mirror(rivets("Pauldron_Rivets", sh, [(sx, sy) for sx in (-0.071, 0.071) for sy in (-0.089, 0.089)], 0.0145, mat=M["metal"], coll=C))

    # elbow and knee
    elbow = OnLoft(ARM, D(300), T_ELBOW)
    mirror(plate("Elbow_Pad", elbow, 0.048, 0.060, n=3.2, offset=0.019, thick=0.015, dome=0.006, bevel=0.004, seg=3, mat=arm, coll=C))
    mirror(plate("Elbow_Cap", elbow, 0.028, 0.036, n=3.5, offset=0.029, thick=0.007, dome=0.003, mat=arm, coll=C))
    knee = OnLoft(LEG, FRONT, LEG.t_at_z(0.565))
    mirror(plate("Knee_Pad", knee, 0.066, 0.078, n=3.0, offset=0.013, thick=0.012, dome=0.008, bevel=0.004, seg=3, mat=arm, coll=C))
    mirror(plate("Knee_Cap", knee, 0.042, 0.050, n=3.6, offset=0.024, thick=0.008, dome=0.003, bevel=0.003, seg=3, mat=arm, coll=C))
    mirror(rivets("Knee_Rivets", knee, [(sx, sy) for sx in (-0.050, 0.050) for sy in (-0.058, 0.058)], 0.0145, mat=M["metal"], coll=C))

    # belt, pouches
    tb = BELT.t_at_z
    loft_mesh("Belt", BELT, mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(g, ts=[tb(1.112), tb(1.162)])))
    box("Belt_Buckle", (0.060, 0.016, 0.052), loc=(0, -0.178, 1.137), bevel=0.005, mat=arm, coll=C)
    box("Belt_Buckle_Inset", (0.034, 0.006, 0.026), loc=(0, -0.187, 1.137), bevel=0.002, mat=dark, coll=C)
    for name, phi, size in (("Pouch_Front", D(34), (0.105, 0.132, 0.048)), ("Pouch_Back", D(312), (0.086, 0.094, 0.042))):
        w, h, d = size
        F = kit.anchor_matrix(OnLoft(BELT, phi, tb(1.134)))  # one frame for the whole pouch
        mirror(box(name, size, M=F @ Matrix.Translation((0, 0, d / 2 - 0.006)), bevel=0.012, seg=4, mat=suit, coll=C))
        mirror(box(name + "_Flap", (w + 0.007, h * 0.40, d + 0.006), M=F @ Matrix.Translation((0, h * 0.31, d / 2 - 0.003)),
                   bevel=0.009, seg=4, mat=suit, coll=C))
        mirror(studs(name + "_Snap", [(F @ Vector((0, h * 0.17, d + 0.0005)))[:]], [(F.to_3x3() @ Vector((0, 0, 1)))[:]],
                     r=0.006, mat=M["metal"], coll=C))

    # cargo pockets on the outer thigh
    tl = LEG.t_at_z
    mirror(plate("Cargo_Pocket", OnLoft(LEG, D(22), tl(0.790)), 0.070, 0.100, n=7.0, offset=0.027, thick=0.025, bevel=0.007, seg=3, subsurf=1, mat=suit, coll=C))
    mirror(plate("Cargo_Flap", OnLoft(LEG, D(22), tl(0.866)), 0.075, 0.031, n=6.0, offset=0.034, thick=0.008, bevel=0.003, subsurf=1, mat=suit, coll=C))
    mirror(rivets("Cargo_Snap", OnLoft(LEG, D(22), tl(0.852)), [(0.0, 0.0)], 0.0345, r=0.006, mat=M["metal"], coll=C))

    # collar: shoulder yoke and the helmet's neck ring
    ty = YOKE.t_at_z
    loft_mesh("Collar_Yoke", YOKE, mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(
        g, phis=[FRONT - D(52), FRONT + D(52), BACK - D(50), BACK + D(50)], ts=[ty(1.566)])))
    loft_mesh("Neck_Ring", NECK_RING, mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(g, phis=[FRONT, BACK, OUT, IN])))
    al = np.linspace(0, TAU, 96, endpoint=False)
    tube("Neck_Seal", np.column_stack([0.146 * np.cos(al), -0.018 + 0.149 * np.sin(al), np.full(96, 1.668)]),
         np.tile(Z, (96, 1)), r=0.009, closed=True, mat=M["rubber"], coll=C)
    for sgn in (-1, 1):
        box_at("Collar_Clip", OnLoft(YOKE, FRONT + sgn * D(36), ty(1.588)), (0.030, 0.026, 0.016), sink=0.004, bevel=0.004, mat=dark, coll=C)


# ------------------------------------------------------------------------ helmet
def build_helmet(M):
    C = "Helmet"
    arm, dark = M["armor"], M["dark"]
    tz = lambda z: z - HZ0
    rows = helmet_rows(kit.RES)
    vis = Oval(HELMET, FRONT, tz(1.725), 0.218, 0.075, n=3.4, taper=0.10)

    shell = loft_mesh("Helmet_Shell", HELMET, rows=rows, hole=vis.scaled(0.97), cap1=True, mat=arm, coll=C,
                      attrs=dict(seam=lambda g: seam_attr(
                          g, phis=[BACK, (FRONT, tz(1.835), 9.0)],
                          ts=[(tz(1.800), BACK, D(96)), (tz(1.705), BACK, D(78))])))
    solid(shell, 0.012, bevel=0.003, seg=3)
    loft_mesh("Visor", HELMET, t0=vis.t0 - 0.085, t1=vis.t0 + 0.085, phi0=FRONT - 1.6, phi1=FRONT + 1.6, res=kit.RES * 0.75,
              keep=vis.scaled(1.03), offset=lambda g: -0.011 + 0.019 * (1.0 - np.clip(g.rho, 0, 1) ** 2), mat=M["visor"], coll=C)
    ph, tt = vis.outline(280, 0.985)
    P, N = HELMET.pn(ph, tt, -0.003)
    tube("Visor_Gasket", P, N, r=0.0065, closed=True, mat=M["rubber"], coll=C)

    # layered shell plates
    top = OnEllipsoid(HC, (HA, HB, HCZ), n0=(0.0, 0.20, 1.0), up=(0.0, -1.0, 0.0))
    plate("Helmet_Crown", top, 0.062, 0.118, n=4.0, offset=0.005, thick=0.006, mat=arm, coll=C)
    rear = OnEllipsoid(HC, (HA, HB, HCZ), n0=(0.0, 1.0, 0.25))
    plate("Helmet_Rear", rear, 0.092, 0.046, n=5.0, offset=0.005, thick=0.006, mat=arm, coll=C)
    plate("Helmet_Brow", OnLoft(HELMET, FRONT, tz(1.823)), 0.118, 0.011, n=5.0, offset=0.005, thick=0.007, mat=arm, coll=C)
    plate("Helmet_Nape", OnLoft(HELMET, BACK, tz(1.668)), 0.150, 0.026, n=5.0, offset=0.005, thick=0.007, mat=arm, coll=C)
    mirror(plate("Helmet_Temple", OnLoft(HELMET, D(8), tz(1.800)), 0.050, 0.020, n=5.0, offset=0.003, thick=0.005, mat=arm, coll=C))
    for sgn in (-1, 1):
        plate("Helmet_ChinVent", OnLoft(HELMET, FRONT + sgn * D(17), tz(1.6405)), 0.013, 0.0038, n=4.0, offset=0.001, thick=0.004, mat=dark, coll=C)
    rivets("Helmet_Rivets", top, [(sx, sy) for sx in (-0.047, 0.047) for sy in (-0.10, -0.03, 0.04, 0.10)], 0.0055, r=0.003, mat=M["metal"], coll=C)
    rivets("Helmet_Rivets_Rear", rear, [(sx, sy) for sx in (-0.078, 0.078) for sy in (-0.032, 0.032)], 0.0055, r=0.003, mat=M["metal"], coll=C)

    # ear modules
    ex = HA - 0.004
    mirror(box("Helmet_Ear", (0.026, 0.060, 0.072), loc=(ex, HC[1] + 0.012, 1.715), bevel=0.008, seg=4, mat=arm, coll=C))
    ear = OnPlane((ex + 0.013, HC[1] + 0.012, 1.715), (0, 1, 0), (0, 0, 1))
    mirror(plate("Helmet_Ear_Disc", ear, 0.017, 0.017, n=2.0, offset=0.003, thick=0.004, mat=dark, coll=C))
    mirror(plate("Helmet_Ear_Ring", ear, 0.023, 0.023, n=2.0, offset=0.002, thick=0.003, hole=0.75, mat=arm, coll=C))


# ---------------------------------------------------------------------- backpack
def build_pack(M):
    C = "Pack"
    arm, dark = M["armor"], M["dark"]
    yb, zc = 0.270, 1.4675
    box("Pack", (0.300, 0.155, 0.415), loc=(0, yb - 0.0775, zc), bevel=0.028, seg=4, mat=arm, coll=C)
    back = OnPlane((0, yb, zc), (-1, 0, 0), (0, 0, 1))
    plate("Pack_BackPlate", back, 0.120, 0.180, n=8.0, offset=0.008, thick=0.014, bevel=0.004, seg=3, mat=arm, coll=C)
    plate("Pack_Panel", back, 0.086, 0.046, n=8.0, shift=(0, 0.106), offset=0.011, thick=0.005, mat=M["blue"], coll=C)
    rivets("Pack_Panel_Rivets", back, [(sx, 0.106 + sy) for sx in (-0.073, 0.073) for sy in (-0.033, 0.033)], 0.011, r=0.003, mat=M["metal"], coll=C)
    for sx in (-0.060, 0.060):
        plate("Pack_Vent_Frame", back, 0.036, 0.026, n=6.0, shift=(sx, -0.112), offset=0.013, thick=0.007, hole=0.72, mat=arm, coll=C)
        plate("Pack_Vent", back, 0.030, 0.021, n=6.0, shift=(sx, -0.112), offset=0.0085, thick=0.003, mat=dark, coll=C)
        for dy in (-0.010, 0.0, 0.010):
            plate("Pack_Vent_Louver", back, 0.024, 0.0022, n=4.0, shift=(sx, -0.112 + dy), offset=0.0115, thick=0.003, mat=arm, coll=C)
    rivets("Pack_Rivets", back, [(sx, sy) for sx in (-0.104, 0.104) for sy in (-0.155, 0.0, 0.155)], 0.008, mat=M["metal"], coll=C)
    plate("Pack_Seam_Bar", back, 0.100, 0.004, n=6.0, shift=(0, 0.035), offset=0.0105, thick=0.004, mat=arm, coll=C)
    # sides: latches and vent slots (left side, mirrored)
    side = OnPlane((0.150, yb - 0.075, zc), (0, 1, 0), (0, 0, 1))
    for dz in (-0.085, 0.060):
        mirror(plate("Pack_Latch", side, 0.016, 0.030, n=5.0, shift=(0.030, dz), offset=0.008, thick=0.009, mat=arm, coll=C))
        mirror(plate("Pack_Latch_Pin", side, 0.008, 0.008, n=2.0, shift=(0.030, dz), offset=0.011, thick=0.004, mat=dark, coll=C))
    for dz in (0.135, 0.010):
        mirror(plate("Pack_Side_Slot", side, 0.024, 0.016, n=5.0, shift=(-0.022, dz), offset=0.002, thick=0.003, mat=dark, coll=C))
        mirror(plate("Pack_Side_Slot_Frame", side, 0.029, 0.021, n=5.0, shift=(-0.022, dz), offset=0.005, thick=0.006, hole=0.78, mat=arm, coll=C))
    # top: valve block and carry handle
    box("Pack_Top", (0.120, 0.070, 0.024), loc=(0, yb - 0.070, zc + 0.205), bevel=0.008, mat=arm, coll=C)
    al = np.linspace(0, math.pi, 24)
    tube("Pack_Handle", np.column_stack([0.036 * np.cos(al), np.full(24, yb - 0.070), zc + 0.215 + 0.020 * np.sin(al)]),
         np.tile([0.0, 1.0, 0.0], (24, 1)), r=0.0045, mat=M["rubber"], coll=C)
    # hoses to the suit's back
    for sgn in (-1, 1):
        s = np.linspace(0, 1, 30)
        pts = np.column_stack([sgn * (0.105 + 0.035 * np.sin(math.pi * s)), 0.175 + 0.06 * np.sin(math.pi * s) - 0.03 * s, zc - 0.195 - 0.090 * s])
        tube("Pack_Hose", pts, r=0.011, mat=M["rubber"], coll=C)


# ------------------------------------------------------------------------ gloves
WRIST = np.array([0.447, -0.037, 0.926])
HAND_L = unit([0.19, -0.08, -1.0])  # down the fingers
_hb = np.array([0.86, -0.50, 0.0])
HAND_B = unit(_hb - (_hb @ HAND_L) * HAND_L)  # out of the back of the hand
HAND_W = np.cross(HAND_L, HAND_B)  # towards the thumb


def _digit(name, base, d0, length, r, curl, M, coll):
    """A gloved finger: three phalanges, each bent `curl[i]` further towards the palm."""
    pts, rad = [base - 0.016 * d0, base.copy()], [r, 1.05 * r]
    p, ang, d = base.copy(), 0.0, d0
    for seg, c, k in zip(np.array([0.46, 0.30, 0.24]) * length, curl, (1.0, 0.94, 0.82)):
        ang += c
        d = unit(d0 * math.cos(ang) - HAND_B * math.sin(ang))
        p = p + seg * d
        pts.append(p.copy())
        rad.append(k * r)
    pts += [p + 0.45 * r * d, p + 0.72 * r * d]
    rad += [0.58 * r, 0.16 * r]
    loft = Loft([dict(p=q, a=a, b=a * 0.96, n=2.0) for q, a in zip(pts, rad)], front=HAND_B)
    return mirror(loft_mesh(name, loft, res=0.0022, cap1=True, mat=M["rubber"], coll=coll))


def build_gloves(M):
    C = "Gloves"
    hp = lambda l, w=0.0, b=0.0: WRIST + l * HAND_L + w * HAND_W + b * HAND_B
    palm = Loft([dict(p=hp(l), a=a, b=b, n=2.6) for l, a, b in (
        (-0.020, .040, .034), (0.012, .042, .028), (0.042, .048, .023), (0.072, .050, .020), (0.092, .048, .017), (0.102, .040, .010))],
        front=HAND_B)
    # Loft rings run wrist -> knuckles, so its tangent points DOWN the hand and
    # phi = 0 is the thumb side, 90 deg the back of the hand.
    mirror(loft_mesh("Glove_Palm", palm, res=0.003, cap1=True, mat=M["rubber"], coll=C))
    for name, w, k, length, r, curl in (
            ("Glove_Index", 0.033, 0.07, 0.068, 0.0116, (0.20, 0.35, 0.30)),
            ("Glove_Middle", 0.011, 0.01, 0.075, 0.0120, (0.26, 0.42, 0.32)),
            ("Glove_Ring", -0.011, -0.05, 0.070, 0.0114, (0.30, 0.45, 0.32)),
            ("Glove_Pinky", -0.033, -0.11, 0.055, 0.0102, (0.34, 0.48, 0.32))):
        _digit(name, hp(0.094, w, -0.002), unit(HAND_L + k * HAND_W), length, r, curl, M, C)
    _digit("Glove_Thumb", hp(0.028, 0.038, -0.010), unit(0.62 * HAND_L + 0.70 * HAND_W - 0.25 * HAND_B), 0.072, 0.0142, (0.0, 0.25, 0.22), M, C)
    back = OnLoft(palm, FRONT, 0.070)
    mirror(plate("Glove_Plate", back, 0.035, 0.033, n=4.0, offset=0.006, thick=0.008, bevel=0.002, seg=3, mat=M["armor"], coll=C))
    mirror(plate("Glove_Knuckle", back, 0.040, 0.009, n=5.0, shift=(0, 0.044), offset=0.005, thick=0.006, mat=M["armor"], coll=C))
    # cuff ring on the forearm's end
    f = unit(ARM.Pk[1] - ARM.Pk[0])  # up the forearm
    cuff = Loft([dict(p=WRIST + l * f, a=a, b=a + 0.004, n=2.0) for l, a in (
        (-0.022, .050), (-0.016, .063), (0.010, .066), (0.036, .065), (0.044, .054))])
    mirror(loft_mesh("Glove_Cuff", cuff, mat=M["armor"], coll=C))
    mirror(band_on("Glove_Cuff_Strap", cuff, 0.026, 0.044, offset=0.003, thick=0.004, mat=M["strap"], coll=C))


# ------------------------------------------------------------------------- boots
ANKLE = np.array([0.247, 0.048, 0.0])
_c, _s = math.cos(D(13)), math.sin(D(13))  # toe-out
BX, BY = np.array([_c, _s, 0.0]), np.array([-_s, _c, 0.0])  # boot-local x (outer side), y (towards the heel)
Y_TOE, Y_HEEL = -0.247, 0.108


def bw(x, y, z):
    return ANKLE + x * BX + y * BY + z * Z


def sole_top(y):
    return 0.028 + 0.016 * (y - Y_TOE) / (Y_HEEL - Y_TOE)


FOOT_PROFILE = (  # y', half-width, height of the upper over its base
    (-0.247, .004, .020), (-0.243, .030, .050), (-0.228, .050, .072), (-0.200, .064, .084), (-0.150, .071, .092),
    (-0.100, .070, .102), (-0.050, .066, .120), (-0.010, .064, .140), (0.030, .062, .150), (0.070, .058, .145),
    (0.095, .045, .120), (0.105, .020, .090), (0.108, .004, .050))
FOOT = Loft([dict(p=bw(0, y, sole_top(y) + 0.010), a=a, bf=h, bb=0.014, n=(2.3, 7.0)) for y, a, h in FOOT_PROFILE], front=Z)
SOLE = Loft([dict(p=bw(0, y, sole_top(y) / 2), a=a + 0.007, bf=sole_top(y) / 2 + 0.004, bb=sole_top(y) / 2, n=7.0) for y, a, _ in FOOT_PROFILE], front=Z)
SHAFT = Loft([dict(p=bw(0, -0.004, z), a=a, bf=bf, bb=bb, n=2.2) for z, a, bf, bb in (
    (0.060, .058, .090, .094), (0.110, .060, .086, .090), (0.170, .064, .085, .086), (0.230, .067, .086, .086), (0.275, .066, .086, .086))],
    front=-BY)


def build_boots(M):
    C = "Boots"
    arm, strap = M["armor"], M["strap"]
    ft = lambda y: y - Y_TOE  # arclength along the foot lofts

    def tread(g):
        y = g.t + Y_TOE
        sn = np.sin(g.phi)
        lugs = np.clip((0.5 - np.abs(sn)) / 0.15, 0, 1) * (np.clip(np.sin(TAU * g.t / 0.030) * 3.0, -1, 1) * 0.5 + 0.5)
        arch = np.clip((-sn - 0.6) / 0.2, 0, 1) * np.clip(1.6 * (1.0 - np.abs(y + 0.012) / 0.048), 0, 1)
        return -0.004 * lugs - 0.013 * arch

    mirror(loft_mesh("Boot_Sole", SOLE, res=0.003, disp=tread, cap0=True, cap1=True, mat=M["sole"], coll=C))
    rand = Loft([dict(p=bw(0, y, sole_top(y) + 0.006), a=a + 0.0045, bf=0.011, bb=0.010, n=6.0) for y, a, _ in FOOT_PROFILE], front=Z)
    mirror(loft_mesh("Boot_Rand", rand, cap0=True, cap1=True, mat=M["rubber"], coll=C))
    mirror(loft_mesh("Boot_Foot", FOOT, cap0=True, cap1=True, mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(g, ts=[ft(-0.165), ft(-0.02)]))))
    mirror(loft_mesh("Boot_Shaft", SHAFT, mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(g, phis=[BACK, D(40), D(140)]))))
    top = lambda y: OnLoft(FOOT, FRONT, ft(y))
    mirror(plate("Boot_ToeCap", top(-0.203), 0.088, 0.036, n=3.0, offset=0.005, thick=0.007, bevel=0.002, mat=arm, coll=C))
    for y in (-0.128, -0.062):
        mirror(plate("Boot_Strap", top(y), 0.112, 0.014, n=9.0, offset=0.005, thick=0.005, mat=strap, coll=C))
        mirror(box_at("Boot_Strap_Buckle", top(y), (0.020, 0.024, 0.009), shift=(-0.060, 0.0), sink=-0.004, bevel=0.002, mat=M["metal"], coll=C))
    sz = SHAFT.t_at_z
    mirror(band_on("Boot_Ankle_Strap", SHAFT, sz(0.150), sz(0.184), offset=0.005, thick=0.006, mat=strap, coll=C))
    mirror(box_at("Boot_Ankle_Buckle", OnLoft(SHAFT, D(28), sz(0.167)), (0.026, 0.028, 0.007), sink=-0.003, bevel=0.002, mat=arm, coll=C))
    mirror(plate("Boot_Tongue", OnLoft(SHAFT, FRONT, sz(0.225)), 0.040, 0.034, n=4.0, offset=0.006, thick=0.008, mat=arm, coll=C))
    mirror(plate("Boot_Heel", OnLoft(SHAFT, BACK, sz(0.098)), 0.066, 0.036, n=4.0, offset=0.006, thick=0.008, mat=arm, coll=C))
    mirror(plate("Boot_Ankle_Disc", OnLoft(SHAFT, OUT, sz(0.115)), 0.019, 0.019, n=2.0, offset=0.006, thick=0.006, mat=arm, coll=C))


def build(M):
    build_suit(M)
    build_armor(M)
    build_helmet(M)
    build_pack(M)
    build_gloves(M)
    build_boots(M)
