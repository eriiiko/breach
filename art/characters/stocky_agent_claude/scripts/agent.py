"""The stocky agent, after `art/characters/fem-fat-agent01.JPG`.

Character only: the artwork's shoulder bag (and its strap across the chest) and the
weapon are props and are not modelled. The artwork is one small front view, so the
back is this model's own design, and its lettering is not reproduced.
Heavy-set, strong build, 1.68 m to the helmet top, relaxed A-pose with empty hands.
Faces -Y, +X is the character's left; limbs are modelled on the left and mirrored.
On a left limb `phi` reads 0 = outer side, 90 deg = front, 180 = inner, 270 = back.
"""
import math

import numpy as np
from mathutils import Matrix

import kit
import parts
from kit import (TAU, Loft, OnEllipsoid, OnLoft, OnPlane, Oval, band_mask, band_on, box, box_at, flip_x, fold_field, fold_set,
                 loft_mesh, mirror, plate, seam_attr, solid, tape_attr, tube, unit, wrap)
from parts import CLAMP, ring, rivets
from studio import BLACK, mat_armor, mat_cloth, mat_plain, mat_visor, rgb

D = math.radians
OUT, FRONT, IN, BACK = 0.0, D(90), D(180), D(270)
Z = np.array([0.0, 0.0, 1.0])
HEIGHT = 1.68
BACKDROP = (0.235, 0.216, 0.456)  # the artwork's violet
ORANGE = (0.72, 0.19, 0.03)


def make_materials():
    return dict(
        suit=mat_cloth("suit_charcoal", rgb(0.020, 0.019, 0.025), rgb(0.030, 0.029, 0.036), flex_col=rgb(0.010, 0.010, 0.012),
                       grime=rgb(0.085, 0.078, 0.070), seam_col=(0.008, 0.008, 0.009), crumple=0.45, bump=(0.4, 0.004),
                       rough=(0.76, 0.6), sheen=0.14, stains=0.14, crease_dirt=0.0, dust=0.30, tape=ORANGE, tape_w=0.003),
        armor=mat_armor("armor_white", rgb(0.46, 0.46, 0.47), rgb(0.56, 0.56, 0.57), rough=(0.34, 0.56), lift=1.0,
                        grime=rgb(0.17, 0.15, 0.13), wear_col=(0.14, 0.135, 0.13), wear=0.55, stains=0.34),
        gunmetal=mat_armor("armor_dark", rgb(0.045, 0.046, 0.052), rgb(0.065, 0.066, 0.074), rough=(0.40, 0.62), lift=1.0,
                           grime=rgb(0.09, 0.08, 0.07), wear_col=(0.20, 0.20, 0.21), wear=0.5, stains=0.2),
        glove=mat_plain("glove_pale", rgb(0.36, 0.36, 0.38), 0.5, bump_scale=420.0, bump=0.2, sheen=0.2),
        rubber=mat_plain("seal_rubber", BLACK, 0.55, bump_scale=420.0, bump=0.25, sheen=0.3),
        strap=mat_plain("strap_webbing", rgb(0.020, 0.020, 0.024), 0.72, bump_scale=900.0, bump=0.3),
        sole=mat_plain("boot_sole", rgb(0.018, 0.018, 0.020), 0.75, bump_scale=300.0, bump=0.2),
        dark=mat_plain("dark_fitting", rgb(0.012, 0.012, 0.014), 0.45),
        orange=mat_plain("marker_orange", rgb(*ORANGE), 0.5),
        lens=mat_plain("lens_blue", rgb(0.02, 0.16, 0.62), 0.12, coat=1.0),
        visor=mat_visor("visor_black", (0.004, 0.004, 0.005), (0.020, 0.021, 0.026)),
    )


# ----------------------------------------------------------------- body surfaces
# One welded surface from ankle to neck (see the sleek agent): thick thighs that meet
# high, wide hips, a full waist, a deep chest.
BODY = Loft([ring(*r) for r in (
    (0.085, .170, .020, .046, .050, .052, 2.0, 0), (0.200, .165, .028, .056, .056, .066, 2.0, 0),
    (0.320, .156, .032, .068, .062, .082, 2.0, 0), (0.400, .148, .024, .064, .064, .072, 2.0, 0),
    (0.460, .142, .010, .068, .072, .066, 2.0, 0), (0.540, .134, .000, .084, .088, .082, 2.0, 0),
    (0.640, .124, -.004, .108, .108, .108, 2.0, 0), (0.720, .116, -.004, .118, .116, .122, 2.1, 0),
    (0.760, .110, -.002, .124, .120, .134, 2.2, 1), (0.810, .092, .000, .142, .126, .150, 2.3, 1),
    (0.870, .066, .004, .164, .134, .156, 2.3, 1), (0.940, .030, .004, .186, .138, .140, 2.4, 1),
    (1.000, .000, .002, .190, .140, .126, 2.4, 1), (1.080, .000, .000, .162, .136, .116, 2.4, 1),
    (1.160, .000, .000, .160, .142, .116, 2.4, 1), (1.250, .000, .000, .170, .166, .124, 2.5, 1),
    (1.320, .000, .000, .172, .148, .124, 2.6, 1), (1.370, .000, .000, .162, .114, .114, 2.6, 1),
    (1.410, .000, .000, .120, .086, .094, 2.4, 1), (1.440, .000, -.004, .078, .068, .074, 2.2, 1),
    (1.470, .000, -.008, .062, .060, .064, 2.0, 1), (1.530, .000, -.008, .058, .058, .060, 2.0, 1))])
bz = BODY.t_at_z
CROTCH_Z = 0.735

ARM = Loft([dict(p=p, a=a, b=b, n=2.0) for p, a, b in (
    ((.292, -.020, 0.870), .032, .028), ((.284, -.012, 0.940), .038, .035), ((.268, .004, 1.030), .046, .043),
    ((.252, .016, 1.075), .044, .043), ((.247, .020, 1.090), .043, .043), ((.234, .016, 1.150), .046, .046),
    ((.210, .008, 1.250), .052, .052), ((.186, .002, 1.320), .054, .054), ((.166, .000, 1.360), .054, .058),
    ((.132, .000, 1.385), .050, .056), ((.095, .000, 1.390), .044, .052))])
az = ARM.t_at_z
T_ELBOW = ARM.t_ring(4)


def body_band(name, z0, z1, offset=0.004, thick=0.004, mat=None, coll="Gear", **kw):
    return mirror(band_on(name, BODY, bz(z0), bz(z1), offset=offset, thick=thick, mat=mat, coll=coll, **CLAMP, **kw), merge=True)


# ------------------------------------------------------------------------- suit
def build_suit(M):
    C = "Suit"
    rng = np.random.default_rng(15)
    ff = fold_field(
        fold_set(rng, 8, (bz(0.76), bz(0.90)), (D(35), D(150)), (-0.5, 0.15), (0.04, 0.08), (0.005, 0.008), (0.002, 0.004))  # hip crease
        + fold_set(rng, 6, (bz(0.40), bz(0.54)), (D(215), D(325)), (0.0, 0.2), (0.03, 0.06), (0.005, 0.007), (0.002, 0.003))  # knee hollow
        + fold_set(rng, 6, (bz(0.44), bz(0.56)), (D(30), D(150)), (0.0, 0.2), (0.03, 0.06), (0.005, 0.007), (0.0015, 0.0028))  # over the knee
        + fold_set(rng, 10, (bz(1.02), bz(1.17)), (0, TAU), (0.0, 0.15), (0.04, 0.09), (0.005, 0.008), (0.0015, 0.003))  # waist
        + fold_set(rng, 200, (0.0, BODY.L), (0, TAU), (0.0, 0.6), (0.018, 0.045), (0.0035, 0.006), (0.0006, 0.0015)))

    def flex(g):
        z = g.P[..., 2]
        neck = 1.432 - z
        knee = np.maximum.reduce([np.abs(wrap(g.phi - BACK)) * g.r - 0.085, 0.40 - z, z - 0.53])
        return band_mask(g, np.minimum(neck, knee))

    def seam(g):
        z = g.P[..., 2]
        centre = np.where(z > CROTCH_Z, np.maximum(g.P[..., 0], 0.0), 1.0)
        return seam_attr(g, phis=[(OUT, bz(0.12), bz(1.34)), (IN, 0, bz(0.72))], ts=[bz(1.005)], extra=centre)

    piping = [
        [(66, bz(1.175)), (66, bz(1.045))],  # the two short orange lines on the belly
        [(10, bz(0.90)), (10, bz(0.60))],  # down the outer thigh
        [(-66, bz(1.175)), (-66, bz(1.045))],  # and their twins behind
    ]
    body = loft_mesh("Body", BODY, res=kit.RES * 0.85, coll=C, mat=M["suit"],
                     disp=lambda g: ff(g) * (1 - flex(g)) - 0.0015 * flex(g),
                     attrs=dict(flex=flex, seam=seam, tape=lambda g: tape_attr(g, piping)), **CLAMP)
    mirror(body, merge=True)

    rng = np.random.default_rng(19)
    ff_arm = fold_field(
        fold_set(rng, 6, (T_ELBOW - 0.02, T_ELBOW + 0.08), (D(20), D(160)), (0.0, 0.2), (0.025, 0.05), (0.004, 0.006), (0.0015, 0.003))
        + fold_set(rng, 80, (0.0, ARM.t_ring(8)), (0, TAU), (0.0, 0.6), (0.015, 0.035), (0.0035, 0.005), (0.0005, 0.0013)))

    def flex_arm(g):
        z = g.P[..., 2]
        return band_mask(g, np.maximum.reduce([np.abs(wrap(g.phi - FRONT)) * g.r - 0.06, 1.060 - z, z - 1.135]))

    mirror(loft_mesh("Sleeve", ARM, res=kit.RES * 0.85, coll=C, mat=M["suit"],
                     disp=lambda g: ff_arm(g) * (1 - flex_arm(g)) - 0.0015 * flex_arm(g),
                     attrs=dict(flex=flex_arm, seam=lambda g: seam_attr(g, phis=[(BACK, 0, ARM.t_ring(8))]))))
    # the black armband, left arm only
    band_on("Armband", ARM, az(1.185), az(1.240), offset=0.0035, thick=0.0035, mat=M["strap"], coll=C)


# ------------------------------------------------------------------------ armour
def build_armor(M):
    C = "Armor"
    arm, dark, gun, orange = M["armor"], M["dark"], M["gunmetal"], M["orange"]

    # cuirass: a smooth white shell over chest, upper back and shoulder roots, with a rolled lower rim
    t0, t1 = bz(1.206), bz(1.438)

    def shell_off(g):
        return 0.013 + 0.004 * np.exp(-((g.t - t0) / 0.008) ** 2)

    cuirass = loft_mesh("Cuirass", BODY, t0=t0, t1=t1, offset=shell_off, mat=arm, coll=C, **CLAMP,
                        attrs=dict(seam=lambda g: seam_attr(
                            g, phis=[(D(38), t0, bz(1.33)), (D(322), t0, bz(1.33))], ts=[(bz(1.300), FRONT, D(52)), (bz(1.236), BACK, D(60))],
                            extra=np.maximum(g.P[..., 0], 0.0))))
    solid(cuirass, 0.010, bevel=0.003, seg=3)
    mirror(cuirass, merge=True)
    body_band("Collar", 1.428, 1.470, offset=0.012, thick=0.010, mat=arm, coll=C)

    chest = OnLoft(BODY, FRONT, bz(1.290))
    for i, ds in enumerate((0.052, 0.074, 0.096)):  # three vents on the right breast
        plate("Chest_Vent_%d" % i, chest, 0.007, 0.016, n=2.6, shift=(ds, 0.018), offset=0.0145, thick=0.004, mat=dark, coll=C)
    plate("Chest_Marker", chest, 0.030, 0.008, n=6.0, shift=(-0.072, 0.030), offset=0.0142, thick=0.002, bevel=0.0006, mat=orange, coll=C)
    plate("Chest_Port", chest, 0.016, 0.016, n=2.0, shift=(-0.060, -0.030), offset=0.016, thick=0.006, mat=gun, coll=C)
    plate("Chest_Port_Core", chest, 0.008, 0.008, n=2.0, shift=(-0.060, -0.030), offset=0.0175, thick=0.003, mat=dark, coll=C)
    for sgn in (-1, 1):
        plate("Chest_Clasp", chest, 0.008, 0.018, n=5.0, shift=(sgn * 0.140, -0.058), offset=0.017, thick=0.006, mat=gun, coll=C)

    # back unit (this model's design: the artwork does not show the back)
    spine = OnLoft(BODY, BACK, bz(1.300))
    box_at("Back_Unit", spine, (0.170, 0.190, 0.052), sink=-0.006, bevel=0.016, seg=4, mat=arm, coll=C)
    back = OnLoft(BODY, BACK, bz(1.300))
    F = kit.anchor_matrix(back)
    face = F @ Matrix.Translation((0, 0, 0.058))
    for i, dx in enumerate((-0.042, 0.042)):
        box("Back_Vent_%d" % i, (0.050, 0.030, 0.006), M=face @ Matrix.Translation((dx, -0.050, 0)), bevel=0.002, mat=dark, coll=C)
    box("Back_Marker", (0.070, 0.012, 0.004), M=face @ Matrix.Translation((0, 0.060, 0)), bevel=0.001, mat=orange, coll=C)
    box("Back_Hatch", (0.090, 0.060, 0.006), M=face @ Matrix.Translation((0, 0.012, 0)), bevel=0.003, mat=arm, coll=C)

    # pauldrons: big smooth domes
    sh = OnEllipsoid((0.176, 0.000, 1.352), (0.066, 0.072, 0.070), n0=(0.86, 0.0, 0.51))
    mirror(plate("Pauldron", sh, 0.064, 0.074, n=3.0, offset=0.010, thick=0.009, bevel=0.003, seg=3, mat=arm, coll=C))
    mirror(rivets("Pauldron_Rivets", sh, [(-.040, -.050), (.040, -.050)], 0.0105, r=0.0035, mat=dark, coll=C))

    # gauntlets: white forearm shells, flaring towards the elbow
    g0, g1 = az(0.880), az(1.062)

    def gaunt_off(g):
        k = (g.t - g0) / (g1 - g0)
        return 0.005 + 0.004 * k * k

    mirror(solid(loft_mesh("Gauntlet", ARM, t0=g0, t1=g1, offset=gaunt_off, mat=arm, coll=C,
                           attrs=dict(seam=lambda g: seam_attr(g, phis=[IN], ts=[az(0.915)]))), 0.006, bevel=0.002, seg=3))
    mirror(rivets("Gauntlet_Dots", OnLoft(ARM, OUT, az(0.985)), [(0.0, dt) for dt in (-0.045, -0.015, 0.015, 0.045)], 0.0105, r=0.0035,
                  mat=dark, coll=C))

    # knee caps
    knee = OnLoft(BODY, FRONT, bz(0.475))
    mirror(plate("Knee_Cap", knee, 0.056, 0.060, n=2.8, offset=0.016, thick=0.010, dome=0.008, bevel=0.003, seg=3, mat=arm, coll=C))


# -------------------------------------------------------------------------- gear
def build_gear(M):
    C = "Gear"
    body_band("Belt", 0.990, 1.022, offset=0.005, thick=0.005, mat=M["strap"], coll=C)
    box_at("Belt_Buckle", OnLoft(BODY, FRONT, bz(1.006)), (0.046, 0.034, 0.009), sink=-0.007, bevel=0.003, mat=M["armor"], coll=C)
    for nm, phi, size in (("Belt_Pouch", 32, (0.062, 0.066, 0.026)), ("Belt_Pouch_Back", -40, (0.070, 0.056, 0.028))):
        F = kit.anchor_matrix(OnLoft(BODY, D(phi), bz(0.975)))
        w, h, d = size
        mirror(box(nm, size, M=F @ Matrix.Translation((0, 0, d / 2 + 0.005)), bevel=0.007, seg=3, mat=M["gunmetal"], coll=C))
        mirror(box(nm + "_Flap", (w + 0.004, h * 0.42, d + 0.004), M=F @ Matrix.Translation((0, h * 0.30, d / 2 + 0.007)),
                   bevel=0.006, seg=3, mat=M["gunmetal"], coll=C))


# ------------------------------------------------------------------------ helmet
HC = np.array([0.0, -0.004, 1.575])
HA, HB, HCZ, HZ0 = 0.094, 0.108, 0.105, 1.474
HELMET, helmet_rows = parts.helmet_dome(HC, (HA, HB, HCZ), HZ0, p=2.9, flare=0.5, n=3.0)  # boxy, flat-topped


def build_helmet(M):
    C = "Helmet"
    arm, dark, gun = M["armor"], M["dark"], M["gunmetal"]
    tz = lambda z: z - HZ0
    face = Oval(HELMET, FRONT, tz(1.552), 0.078, 0.052, n=3.4)
    shell = loft_mesh("Helmet_Shell", HELMET, rows=helmet_rows(kit.RES * 0.7), res=kit.RES * 0.7, hole=face.scaled(0.97), cap1=True,
                      mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(
                          g, phis=[BACK, (D(0), 0, tz(1.60)), (D(180), 0, tz(1.60))], ts=[(tz(1.612), BACK, D(120)), (tz(1.520), BACK, D(75))])))
    solid(shell, 0.009, bevel=0.0025, seg=3)
    # the recessed dark face under the shell's opening
    loft_mesh("Face_Mask", HELMET, t0=face.t0 - 0.066, t1=face.t0 + 0.066, phi0=FRONT - 1.0, phi1=FRONT + 1.0, res=kit.RES * 0.7,
              keep=face.scaled(1.04), offset=-0.008, mat=gun, coll=C)
    ph, tt = face.outline(200, 0.985)
    P, N = HELMET.pn(ph, tt, -0.003)
    tube("Face_Gasket", P, N, r=0.0036, closed=True, mat=M["rubber"], coll=C)

    yf = HC[1] - HB  # the helmet's front
    # visor housing: a white shroud across the brow with a dark slit and one blue lens
    box("Visor_Housing", (0.150, 0.046, 0.044), loc=(0, yf + 0.006, 1.606), rot=(D(90), 0, 0), taper=(0.92, 0.86), bevel=0.009, seg=4,
        mat=arm, coll=C)
    box("Visor_Slit", (0.064, 0.006, 0.012), loc=(0.018, yf - 0.017, 1.607), bevel=0.002, mat=M["visor"], coll=C)
    lens = OnPlane((-0.044, yf - 0.016, 1.607), (-1, 0, 0), (0, 0, 1))
    plate("Visor_Lens_Ring", lens, 0.012, 0.012, n=2.0, offset=0.004, thick=0.006, mat=gun, coll=C)
    plate("Visor_Lens", lens, 0.008, 0.008, n=2.0, offset=0.0055, thick=0.003, dome=0.002, mat=M["lens"], coll=C)
    # respirator
    box("Respirator", (0.058, 0.040, 0.040), loc=(0, yf + 0.010, 1.527), rot=(D(90), 0, 0), taper=(0.70, 0.74), bevel=0.007, seg=3,
        mat=gun, coll=C)
    for i, dz in enumerate((-0.008, 0.004)):
        box("Respirator_Vent_%d" % i, (0.026, 0.004, 0.0035), loc=(0, yf - 0.0105, 1.527 + dz), bevel=0.001, mat=dark, coll=C)
    cap = OnPlane((0.036, yf + 0.016, 1.524), (0.66, 0.75, 0.0), (0.0, 0.0, 1.0))
    mirror(plate("Respirator_Filter", cap, 0.014, 0.014, n=2.0, offset=0.010, thick=0.012, bevel=0.002, mat=gun, coll=C))

    # ear housings and a low crest
    ear = OnPlane((HA - 0.006, HC[1] + 0.010, 1.566), (0, 1, 0), (0, 0, 1))
    mirror(plate("Ear_Housing", ear, 0.030, 0.036, n=4.0, offset=0.014, thick=0.016, bevel=0.004, seg=3, mat=arm, coll=C))
    mirror(plate("Ear_Core", ear, 0.012, 0.012, n=2.0, offset=0.0155, thick=0.003, mat=dark, coll=C))
    crown = OnEllipsoid(HC, (HA, HB, HCZ), n0=(0.0, 0.15, 1.0), up=(0.0, -1.0, 0.0))
    plate("Helmet_Crest", crown, 0.026, 0.070, n=5.0, offset=0.004, thick=0.006, mat=arm, coll=C)
    plate("Helmet_Nape", OnLoft(HELMET, BACK, tz(1.508)), 0.066, 0.020, n=5.0, offset=0.004, thick=0.006, mat=arm, coll=C)
    plate("Helmet_Marker", OnLoft(HELMET, D(300), tz(1.585)), 0.020, 0.006, n=6.0, offset=0.0012, thick=0.002, bevel=0.0005,
          mat=M["orange"], coll=C)

    seal = Loft([ring(z, 0.0, cy, a, bf, bb, 2.2) for z, cy, a, bf, bb in (
        (1.440, -.008, .064, .064, .068), (1.458, -.006, .070, .076, .080), (1.478, -.004, .082, .094, .096))])
    loft_mesh("Neck_Seal", seal, mat=M["rubber"], coll=C)


# ------------------------------------------------------------------------ gloves
WRIST = np.array([0.292, -0.020, 0.870])
HAND_L = unit([0.14, -0.06, -1.0])
_hb = np.array([0.88, -0.47, 0.0])
HAND_B = unit(_hb - (_hb @ HAND_L) * HAND_L)


def build_gloves(M):
    C = "Gloves"
    k = 0.86
    palm, _ = parts.hand("Glove", WRIST, HAND_L, HAND_B, M["glove"], C, scale=k, curl=1.05)
    back = OnLoft(palm, FRONT, 0.064 * k)
    mirror(plate("Glove_Pad", back, 0.030 * k, 0.026 * k, n=4.0, offset=0.004, thick=0.005, bevel=0.0015, seg=3, mat=M["armor"], coll=C))
    f = unit(ARM.Pk[1] - ARM.Pk[0])
    cuff = Loft([dict(p=WRIST + l * f, a=a, b=a - 0.004, n=2.0) for l, a in ((-0.022, .032), (-0.015, .039), (0.004, .040), (0.020, .039), (0.026, .034))])
    mirror(loft_mesh("Glove_Cuff", cuff, mat=M["strap"], coll=C))


# ------------------------------------------------------------------------- boots
ANKLE = np.array([0.172, 0.024, 0.0])
_c, _s = math.cos(D(10)), math.sin(D(10))  # toe-out
BX, BY = np.array([_c, _s, 0.0]), np.array([-_s, _c, 0.0])
Y_TOE, Y_HEEL = -0.185, 0.078


def bw(x, y, z):
    return ANKLE + x * BX + y * BY + z * Z


def sole_top(y):
    return 0.026 + 0.012 * (y - Y_TOE) / (Y_HEEL - Y_TOE)


FOOT_PROFILE = (  # y', half-width, height of the upper over its base
    (-0.185, .004, .014), (-0.181, .026, .032), (-0.166, .043, .046), (-0.140, .052, .054), (-0.100, .056, .060),
    (-0.060, .054, .070), (-0.025, .052, .090), (0.005, .050, .110), (0.035, .048, .118), (0.060, .043, .108),
    (0.071, .031, .082), (0.077, .014, .052), (0.078, .004, .030))
FOOT = Loft([dict(p=bw(0, y, sole_top(y) + 0.008), a=a, bf=h, bb=0.012, n=(2.5, 7.0)) for y, a, h in FOOT_PROFILE], front=Z)
SOLE = Loft([dict(p=bw(0, y, sole_top(y) / 2), a=a + 0.006, bf=sole_top(y) / 2 + 0.004, bb=sole_top(y) / 2, n=7.0) for y, a, _ in FOOT_PROFILE], front=Z)


def build_boots(M):
    C = "Boots"
    arm = M["armor"]
    ft = lambda y: y - Y_TOE
    mirror(loft_mesh("Boot_Sole", SOLE, res=0.0028, disp=parts.tread(Y_TOE, pitch=0.026, arch_y=-0.018, arch_w=0.038), cap0=True, cap1=True,
                     mat=M["sole"], coll=C))
    mirror(loft_mesh("Boot_Foot", FOOT, res=0.003, cap0=True, cap1=True, mat=arm, coll=C,
                     attrs=dict(seam=lambda g: seam_attr(g, ts=[ft(-0.125), ft(-0.035)]))))
    # knee-high shaft: a white shell on the lower leg with a rolled top rim
    s0, s1 = bz(0.090), bz(0.428)

    def shaft_off(g):
        return 0.009 + 0.005 * np.exp(-((g.t - s1) / 0.010) ** 2)

    mirror(solid(loft_mesh("Boot_Shaft", BODY, t0=s0, t1=s1, offset=shaft_off, mat=arm, coll=C,
                           attrs=dict(seam=lambda g: seam_attr(g, phis=[BACK], ts=[bz(0.150)]))), 0.007, bevel=0.002, seg=3))
    mirror(rivets("Boot_Dots", OnLoft(BODY, FRONT, bz(0.280)), [(0.0, dt) for dt in (-0.110, -0.070, -0.030, 0.010, 0.050, 0.090)], 0.012,
                  r=0.004, mat=M["dark"], coll=C))
    mirror(band_on("Boot_Strap", BODY, bz(0.118), bz(0.142), offset=0.0125, thick=0.004, mat=M["strap"], coll=C))
    mirror(box_at("Boot_Buckle", OnLoft(BODY, D(15), bz(0.130)), (0.022, 0.020, 0.005), sink=-0.016, bevel=0.0015, mat=M["gunmetal"], coll=C))


def build(M):
    build_suit(M)
    build_armor(M)
    build_gear(M)
    build_helmet(M)
    build_gloves(M)
    build_boots(M)


# Beauty views: (rig azimuth, camera azimuth, elevation, distance, target height, lens mm, resolution)
BEAUTY = {
    "hero": (0.0, 30.0, 6.0, 4.9, 0.85, 85.0, (1200, 1600)),
    "hero_back": (180.0, 30.0, 6.0, 4.9, 0.85, 85.0, (1200, 1600)),
    "closeup": (0.0, 22.0, 3.0, 2.3, 1.40, 85.0, (1400, 1400)),
    "closeup_back": (180.0, 24.0, 8.0, 2.3, 1.38, 85.0, (1400, 1400)),
    "legs": (0.0, 26.0, 6.0, 2.7, 0.42, 85.0, (1400, 1400)),
    "elevated": (0.0, 35.0, 52.0, 4.8, 0.85, 85.0, (1200, 1400)),
    "top": (0.0, 0.0, 89.0, 4.2, 0.85, 85.0, (1200, 1200)),
    "art": (0.0, 0.0, 2.0, 5.2, 0.86, 85.0, (900, 1750)),  # the artwork's straight-on view
}
