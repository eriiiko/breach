"""The sleek agent, after `art/characters/sleek-agent-with-bag.png`.

Character only: the artwork's duffel bag and weapon are props and are not modelled
here. The artwork is a single low-angle view, so the back is this model's own design.
Slim athletic build, 1.74 m to the helmet top, relaxed A-pose with empty hands.
Faces -Y, +X is the character's left; limbs are modelled on the left and mirrored.
On a left limb `phi` reads 0 = outer side, 90 deg = front, 180 = inner, 270 = back.
"""
import math

import numpy as np
from mathutils import Matrix

import kit
import parts
from parts import CLAMP, ring, rivets
from kit import (TAU, Loft, OnEllipsoid, OnLoft, OnPlane, Oval, band_mask, band_on, box, box_at, facet_anchor, facet_loft,
                 flip_x, fold_field, fold_set, loft_mesh, mirror, plate, ribbon_on, seam_attr, solid, studs, tape_attr, tube,
                 unit, wrap)
from studio import BLACK, mat_armor, mat_cloth, mat_plain, mat_visor, rgb

D = math.radians
OUT, FRONT, IN, BACK = 0.0, D(90), D(180), D(270)
Z = np.array([0.0, 0.0, 1.0])
HEIGHT = 1.74
BACKDROP = (0.235, 0.216, 0.456)  # the artwork's violet


def make_materials():
    return dict(
        suit=mat_cloth("suit_charcoal", rgb(0.016, 0.017, 0.020), rgb(0.024, 0.025, 0.029), flex_col=rgb(0.009, 0.009, 0.011),
                       grime=rgb(0.085, 0.078, 0.070), seam_col=(0.008, 0.008, 0.009), crumple=0.30, bump=(0.35, 0.003),
                       rough=(0.74, 0.58), sheen=0.12, stains=0.14, crease_dirt=0.0, dust=0.30, tape=(0.105, 0.105, 0.115)),
        armor=mat_armor("armor_pale", rgb(0.42, 0.41, 0.39), rgb(0.52, 0.51, 0.49), rough=(0.36, 0.60), lift=1.0,
                        grime=rgb(0.16, 0.14, 0.12), wear_col=(0.11, 0.105, 0.10), wear=0.9, stains=0.42),
        gunmetal=mat_armor("armor_dark", rgb(0.045, 0.046, 0.050), rgb(0.065, 0.066, 0.072), rough=(0.40, 0.62), lift=1.0,
                           grime=rgb(0.09, 0.08, 0.07), wear_col=(0.20, 0.20, 0.21), wear=0.5, stains=0.2),
        rubber=mat_plain("glove_rubber", BLACK, 0.55, bump_scale=420.0, bump=0.25, sheen=0.3),
        strap=mat_plain("strap_webbing", rgb(0.022, 0.022, 0.025), 0.72, bump_scale=900.0, bump=0.3),
        sole=mat_plain("boot_sole", rgb(0.018, 0.018, 0.020), 0.75, bump_scale=300.0, bump=0.2),
        dark=mat_plain("dark_fitting", rgb(0.012, 0.012, 0.014), 0.45),
        bronze=mat_plain("buckle_bronze", rgb(0.36, 0.22, 0.10), 0.42, metallic=1.0),
        visor=mat_visor("visor_black", (0.004, 0.004, 0.005), (0.020, 0.021, 0.026)),
    )


# ----------------------------------------------------------------- body surfaces
# The bodysuit is ONE surface from ankle to neck: each leg's section grows into half
# the pelvis, then half the torso, clamped at the mid-plane and welded to its mirror.
BODY = Loft([ring(*r) for r in (
    (0.090, .170, .020, .036, .040, .042, 2.0, 0), (0.200, .160, .028, .042, .044, .052, 2.0, 0),
    (0.330, .147, .030, .052, .050, .066, 2.0, 0), (0.430, .135, .020, .048, .050, .056, 2.0, 0),
    (0.500, .127, .008, .052, .056, .052, 2.0, 0), (0.580, .119, .000, .060, .064, .060, 2.0, 0),
    (0.715, .106, -.004, .073, .078, .076, 2.0, 0), (0.820, .095, -.004, .083, .086, .090, 2.1, 0),
    (0.860, .088, -.002, .086, .088, .098, 2.2, 1), (0.900, .073, .000, .100, .090, .108, 2.3, 1),
    (0.940, .053, .002, .112, .092, .112, 2.3, 1), (0.995, .020, .002, .129, .092, .100, 2.3, 1),
    (1.040, .000, .000, .128, .087, .090, 2.2, 1), (1.100, .000, .000, .112, .082, .084, 2.1, 1),
    (1.160, .000, .000, .118, .088, .088, 2.1, 1), (1.215, .000, .000, .128, .100, .092, 2.2, 1),
    (1.280, .000, .000, .140, .120, .096, 2.3, 1), (1.340, .000, .000, .148, .110, .098, 2.4, 1),
    (1.390, .000, .000, .150, .092, .095, 2.5, 1), (1.430, .000, .000, .118, .075, .082, 2.4, 1),
    (1.465, .000, -.004, .070, .058, .064, 2.2, 1), (1.500, .000, -.008, .054, .054, .058, 2.0, 1),
    (1.560, .000, -.008, .052, .052, .056, 2.0, 1))])
bz = BODY.t_at_z
CROTCH_Z = 0.85

ARM = Loft([dict(p=p, a=a, b=b, n=2.0) for p, a, b in (
    ((.265, -.020, 0.895), .026, .022), ((.258, -.012, 0.960), .031, .028), ((.243, .004, 1.050), .037, .034),
    ((.230, .013, 1.110), .035, .033), ((.225, .015, 1.135), .034, .033), ((.215, .012, 1.190), .035, .035),
    ((.195, .006, 1.290), .039, .039), ((.175, .002, 1.360), .042, .042), ((.160, .000, 1.405), .044, .047),
    ((.130, .000, 1.425), .042, .047), ((.095, .000, 1.430), .036, .044))])
az = ARM.t_at_z
T_ELBOW = ARM.t_ring(4)

def body_band(name, z0, z1, offset=0.004, thick=0.004, mat=None, coll="Gear"):
    """A strap right round the pelvis or torso: built on the left half and welded to its mirror."""
    return mirror(band_on(name, BODY, bz(z0), bz(z1), offset=offset, thick=thick, mat=mat, coll=coll, **CLAMP), merge=True)


# ------------------------------------------------------------------------- suit
def build_suit(M):
    C = "Suit"
    rng = np.random.default_rng(5)
    ff = fold_field(
        fold_set(rng, 6, (bz(0.86), bz(0.95)), (D(35), D(150)), (-0.5, 0.15), (0.03, 0.06), (0.004, 0.006), (0.0015, 0.003))  # hip crease
        + fold_set(rng, 5, (bz(0.43), bz(0.56)), (D(215), D(325)), (0.0, 0.2), (0.025, 0.05), (0.004, 0.006), (0.0015, 0.0025))  # knee hollow
        + fold_set(rng, 8, (bz(1.04), bz(1.16)), (0, TAU), (0.0, 0.15), (0.03, 0.07), (0.004, 0.006), (0.001, 0.0022))  # waist
        + fold_set(rng, 170, (0.0, BODY.L), (0, TAU), (0.0, 0.6), (0.015, 0.04), (0.003, 0.005), (0.0004, 0.0010)))

    def flex(g):
        z = g.P[..., 2]
        neck = 1.458 - z
        knee = np.maximum.reduce([np.abs(wrap(g.phi - BACK)) * g.r - 0.07, 0.43 - z, z - 0.57])
        side = np.minimum(np.abs(wrap(g.phi - OUT)), np.abs(wrap(g.phi - IN))) * g.r
        waist = np.maximum.reduce([side - 0.040, 1.03 - z, z - 1.19])
        return band_mask(g, np.minimum.reduce([neck, knee, waist]))

    def seam(g):
        z = g.P[..., 2]
        centre = np.where(z > CROTCH_Z, np.maximum(g.P[..., 0], 0.0), 1.0)
        return seam_attr(g, phis=[(OUT, bz(0.12), bz(1.36)), (IN, 0, bz(0.80))], extra=centre)

    tape = [
        [(15, bz(0.91)), (55, bz(0.80)), (110, bz(0.67)), (150, bz(0.585))],  # hip to inner knee, across the front
        [(-20, bz(0.93)), (-8, bz(0.75)), (8, bz(0.60))],  # down the outer thigh
        [(-60, bz(0.88)), (-100, bz(0.76)), (-140, bz(0.66)), (-165, bz(0.59))],  # the same sweep behind
        [(38, bz(1.205)), (27, bz(1.10)), (36, bz(1.00)), (20, bz(0.93))],  # princess seam, front
        [(-38, bz(1.205)), (-27, bz(1.10)), (-40, bz(0.99)), (-25, bz(0.93))],  # and back
    ]
    body = loft_mesh("Body", BODY, res=kit.RES * 0.85, coll=C, mat=M["suit"],
                     disp=lambda g: ff(g) * (1 - flex(g)) - 0.0015 * flex(g),
                     attrs=dict(flex=flex, seam=seam, tape=lambda g: tape_attr(g, tape)), **CLAMP)
    mirror(body, merge=True)

    rng = np.random.default_rng(9)
    ff_arm = fold_field(
        fold_set(rng, 5, (T_ELBOW - 0.05, T_ELBOW + 0.05), (D(20), D(160)), (0.0, 0.2), (0.02, 0.04), (0.003, 0.005), (0.001, 0.002))
        + fold_set(rng, 70, (0.0, ARM.t_ring(8)), (0, TAU), (0.0, 0.6), (0.012, 0.03), (0.003, 0.0045), (0.0004, 0.0010)))

    def flex_arm(g):
        z = g.P[..., 2]
        return band_mask(g, np.maximum.reduce([np.abs(wrap(g.phi - FRONT)) * g.r - 0.045, 1.100 - z, z - 1.170]))

    mirror(loft_mesh("Sleeve", ARM, res=kit.RES * 0.85, coll=C, mat=M["suit"],
                     disp=lambda g: ff_arm(g) * (1 - flex_arm(g)) - 0.0015 * flex_arm(g),
                     attrs=dict(flex=flex_arm, seam=lambda g: seam_attr(g, phis=[(BACK, 0, ARM.t_ring(8))]),
                                tape=lambda g: tape_attr(g, [[(-60, az(1.37)), (0, az(1.33)), (60, az(1.37))]]))))
    # upper-arm patch pocket, right arm only
    flip_x(plate("Sleeve_Patch", OnLoft(ARM, D(25), az(1.285)), 0.020, 0.030, n=8.0, offset=0.004, thick=0.004, bevel=0.001,
                 mat=M["strap"], coll=C))


# ------------------------------------------------------------------------ armour
def build_armor(M):
    C = "Armor"
    arm, dark, strap, bronze = M["armor"], M["dark"], M["strap"], M["bronze"]

    # breastplate: a faceted prow over the bust, notched lower edge (left half, welded to its mirror)
    def chest_row(z, cols):
        return (bz(z), [(90.0 - d, o) for d, o in cols])

    mirror(facet_loft("Breastplate", BODY, [
        chest_row(1.190, [(0, .013), (16, .012), (27, .010), (34, .009)]),
        chest_row(1.236, [(0, .022), (20, .020), (42, .011), (56, .008)]),
        chest_row(1.292, [(0, .024), (19, .022), (47, .012), (66, .008)]),
        chest_row(1.355, [(0, .022), (17, .020), (43, .011), (62, .008)]),
        chest_row(1.406, [(0, .010), (19, .010), (38, .009), (50, .008)]),
    ], thick=0.007, bevel=0.002, mat=arm, coll=C), merge=True)
    chest = OnLoft(BODY, FRONT, bz(1.345))
    box_at("Breast_Slot", chest, (0.007, 0.034, 0.004), sink=-0.021, bevel=0.001, mat=dark, coll=C)
    mirror(rivets("Breast_Rivets", OnLoft(BODY, FRONT, bz(1.29)), [(0.062, 0.085), (0.088, -0.035), (0.050, -0.078)], 0.010, mat=bronze, coll=C))

    # back plate with a raised spine (this model's design: the artwork does not show the back)
    def back_row(z, cols):
        return (bz(z), [(270.0 + d, o) for d, o in cols])

    mirror(facet_loft("Backplate", BODY, [
        back_row(1.205, [(0, .012), (13, .010), (24, .008)]),
        back_row(1.262, [(0, .022), (20, .012), (40, .008)]),
        back_row(1.340, [(0, .022), (20, .012), (44, .008)]),
        back_row(1.402, [(0, .012), (18, .010), (34, .008)]),
    ], thick=0.007, bevel=0.002, mat=arm, coll=C), merge=True)
    spine = OnLoft(BODY, BACK, bz(1.30))
    box_at("Back_Cell", spine, (0.050, 0.085, 0.020), sink=-0.018, bevel=0.005, mat=M["gunmetal"], coll=C)
    box_at("Back_Cell_Cap", spine, (0.030, 0.012, 0.008), shift=(0, 0.050), sink=-0.022, bevel=0.002, mat=dark, coll=C)

    for i, z in enumerate((1.155, 1.105, 1.055)):
        plate("Spine_Segment_%d" % i, OnLoft(BODY, BACK, bz(z)), 0.019, 0.017, n=5.0, offset=0.008, thick=0.006, bevel=0.0015,
              mat=arm, coll=C)

    # harness: webbing over the shoulders and round the ribs, joining the two plates
    mirror(ribbon_on("Harness_Shoulder", BODY, [(52, bz(1.372)), (40, bz(1.412)), (15, bz(1.438)), (-15, bz(1.438)),
                                                (-40, bz(1.412)), (-52, bz(1.372))], 0.028, offset=0.011, mat=strap, coll=C))
    mirror(ribbon_on("Harness_Side", BODY, [(56, bz(1.262)), (15, bz(1.250)), (-15, bz(1.250)), (-40, bz(1.262))], 0.024,
                     offset=0.005, mat=strap, coll=C))
    for nm, phi, z in (("Harness_Buckle_F", 58, 1.262), ("Harness_Buckle_S", 56, 1.386)):
        mirror(box_at(nm, OnLoft(BODY, D(phi), bz(z)), (0.018, 0.026, 0.006), sink=-0.011, bevel=0.0015, mat=bronze, coll=C))
    body_band("Underbust_Strap", 1.176, 1.194, offset=0.004, thick=0.004, mat=strap, coll=C)
    mirror(box_at("Underbust_Buckle", OnLoft(BODY, D(62), bz(1.185)), (0.026, 0.024, 0.007), sink=-0.004, bevel=0.0015, mat=bronze, coll=C))

    # pauldron: an angular cap over the deltoid
    def cap_row(t, spread, lo, hi):
        return (t, [(-spread, lo), (-spread / 3.0, hi), (spread / 3.0, hi), (spread, lo)])

    mirror(facet_loft("Pauldron", ARM, [
        cap_row(az(1.300), 50, .006, .010), cap_row(az(1.345), 70, .007, .014), cap_row(az(1.395), 72, .007, .015),
        cap_row(ARM.t_ring(8) + 0.024, 62, .007, .012), cap_row(ARM.t_ring(9) + 0.004, 40, .006, .008),
    ], thick=0.007, bevel=0.002, mat=arm, coll=C))
    sh = OnLoft(ARM, OUT, az(1.368))
    mirror(box_at("Pauldron_Slot", sh, (0.006, 0.030, 0.004), sink=-0.0155, bevel=0.001, mat=dark, coll=C))
    mirror(rivets("Pauldron_Rivets", sh, [(-.034, -.040), (.034, -.040), (-.036, .030), (.036, .030)], 0.0085, mat=bronze, coll=C))
    mirror(plate("Harness_Pad", OnLoft(BODY, D(50), bz(1.420)), 0.014, 0.020, n=6.0, offset=0.016, thick=0.005, bevel=0.0015,
                 mat=arm, coll=C))

    # collar guard round the front and sides of the neck
    def neck_row(z, o):
        return (bz(z), [(8.0, o * 0.8), (38.0, o), (66.0, o), (90.0, o)])

    mirror(facet_loft("Collar_Ring", BODY, [neck_row(1.448, .006), neck_row(1.466, .009), neck_row(1.482, .011)],
                      thick=0.005, bevel=0.0015, mat=M["gunmetal"], coll=C), merge=True)
    flip_x(facet_loft("Collar_Plate", BODY, [
        (bz(1.446), [(18.0, .012), (44.0, .013), (72.0, .012)]),
        (bz(1.478), [(14.0, .018), (44.0, .019), (76.0, .017)]),
        (bz(1.512), [(20.0, .024), (44.0, .026), (66.0, .023)]),
    ], thick=0.006, bevel=0.0018, mat=arm, coll=C))

    # forearm bracer: outer shell, raised top plate, two straps underneath
    def arm_row(z, spread, offs):
        return (az(z), [(k * spread, o) for k, o in zip((-1.0, -0.5, 0.0, 0.5, 1.0), offs)])

    mirror(facet_loft("Bracer", ARM, [
        arm_row(0.915, 80, (.006, .008, .009, .008, .006)), arm_row(0.965, 96, (.007, .010, .012, .010, .007)),
        arm_row(1.040, 100, (.007, .011, .013, .011, .007)), arm_row(1.096, 80, (.007, .010, .012, .010, .007)),
    ], thick=0.006, bevel=0.0018, mat=arm, coll=C))
    top = OnLoft(ARM, OUT, az(1.005))
    mirror(plate("Bracer_Top", top, 0.017, 0.048, n=6.0, offset=0.0175, thick=0.006, bevel=0.0015, mat=arm, coll=C))
    mirror(box_at("Bracer_Slot", top, (0.005, 0.030, 0.003), sink=-0.0175, bevel=0.001, mat=dark, coll=C))
    for nm, z in (("Bracer_Strap_A", 0.940), ("Bracer_Strap_B", 1.064)):
        mirror(band_on(nm, ARM, az(z), az(z + 0.016), offset=0.0035, thick=0.003, mat=strap, coll=C))
    mirror(plate("Elbow_Cap", OnLoft(ARM, D(290), T_ELBOW), 0.016, 0.022, n=5.0, offset=0.008, thick=0.006, dome=0.001,
                 bevel=0.002, mat=arm, coll=C))

    # greave: pointed knee, ridged shin, wrapping the front and both sides of the lower leg
    def leg_row(z, phis, offs):
        return (bz(z), list(zip(phis, offs)))

    mirror(facet_loft("Greave", BODY, [
        leg_row(0.166, (-15, 42, 90, 138, 195), (.008, .010, .013, .010, .008)),
        leg_row(0.220, (-25, 38, 90, 142, 205), (.008, .010, .014, .010, .008)),
        leg_row(0.360, (-20, 40, 90, 140, 200), (.008, .011, .016, .011, .008)),
        leg_row(0.450, (5, 50, 90, 130, 175), (.009, .014, .018, .014, .009)),
        leg_row(0.502, (25, 60, 90, 120, 155), (.010, .018, .024, .018, .010)),
        leg_row(0.548, (66, 80, 90, 100, 114), (.010, .016, .019, .016, .010)),
    ], thick=0.007, bevel=0.002, mat=arm, coll=C))
    for nm, phi in (("Knee_Hinge_Out", 12), ("Knee_Hinge_In", 168)):
        mirror(plate(nm, OnLoft(BODY, D(phi), bz(0.497)), 0.014, 0.014, n=2.0, offset=0.014, thick=0.007, bevel=0.0015,
                     mat=M["gunmetal"], coll=C))
    for nm, z in (("Greave_Slot_A", 0.300), ("Greave_Slot_B", 0.400)):
        mirror(box_at(nm, OnLoft(BODY, D(48), bz(z)), (0.006, 0.034, 0.004), sink=-0.0125, bevel=0.001, mat=dark, coll=C))
    for nm, z in (("Greave_Strap_A", 0.385), ("Greave_Strap_B", 0.205)):
        mirror(band_on(nm, BODY, bz(z), bz(z + 0.020), phi0=D(195), phi1=D(345), offset=0.004, thick=0.004, mat=strap, coll=C))
    mirror(rivets("Greave_Rivets", OnLoft(BODY, FRONT, bz(0.33)), [(-0.036, 0.10), (0.036, 0.10), (-0.034, -0.12), (0.034, -0.12)],
                  0.0125, mat=bronze, coll=C))


# -------------------------------------------------------------------------- gear
def build_gear(M):
    C = "Gear"
    strap, bronze, gun = M["strap"], M["bronze"], M["gunmetal"]
    body_band("Belt", 1.000, 1.026, offset=0.004, thick=0.005, mat=strap, coll=C)
    box_at("Belt_Buckle", OnLoft(BODY, FRONT, bz(1.013)), (0.034, 0.030, 0.008), sink=-0.006, bevel=0.002, mat=bronze, coll=C)
    for nm, phi, size in (("Belt_Pouch", 28, (0.050, 0.058, 0.022)), ("Belt_Pouch_Back", -42, (0.060, 0.050, 0.024))):
        F = kit.anchor_matrix(OnLoft(BODY, D(phi), bz(0.992)))
        w, h, d = size
        mirror(box(nm, size, M=F @ Matrix.Translation((0, 0, d / 2 + 0.004)), bevel=0.006, seg=3, mat=gun, coll=C))
        mirror(box(nm + "_Flap", (w + 0.004, h * 0.42, d + 0.004), M=F @ Matrix.Translation((0, h * 0.30, d / 2 + 0.006)),
                   bevel=0.005, seg=3, mat=gun, coll=C))
    # thigh rig: two straps and a pouch on the right leg, one strap and a small pouch on the left
    for nm, z in (("Thigh_Strap_A", 0.735), ("Thigh_Strap_B", 0.655)):
        flip_x(band_on(nm, BODY, bz(z), bz(z + 0.022), offset=0.0035, thick=0.0035, mat=strap, coll=C))
    flip_x(box_at("Thigh_Pouch", OnLoft(BODY, D(8), bz(0.707)), (0.058, 0.092, 0.024), sink=-0.003, bevel=0.006, mat=gun, coll=C))
    flip_x(box_at("Thigh_Pouch_Flap", OnLoft(BODY, D(8), bz(0.740)), (0.062, 0.030, 0.028), sink=-0.003, bevel=0.005, mat=gun, coll=C))
    flip_x(box_at("Thigh_Buckle", OnLoft(BODY, D(62), bz(0.746)), (0.022, 0.018, 0.006), sink=-0.006, bevel=0.0015, mat=bronze, coll=C))
    band_on("Thigh_Strap_L", BODY, bz(0.700), bz(0.722), offset=0.0035, thick=0.0035, mat=strap, coll=C)
    box_at("Thigh_Pouch_L", OnLoft(BODY, D(350), bz(0.711)), (0.044, 0.056, 0.020), sink=-0.003, bevel=0.005, mat=gun, coll=C)


# ------------------------------------------------------------------------ helmet
HC = np.array([0.0, -0.004, 1.632])
HA, HB, HCZ, HZ0 = 0.090, 0.108, 0.108, 1.528


HELMET, helmet_rows = parts.helmet_dome(HC, (HA, HB, HCZ), HZ0, p=2.0, flare=0.55, n=2.1)


def build_helmet(M):
    C = "Helmet"
    arm, dark, gun = M["armor"], M["dark"], M["gunmetal"]
    tz = lambda z: z - HZ0
    vis = Oval(HELMET, FRONT, tz(1.6385), 0.143, 0.033, n=3.6, taper=0.12)
    shell = loft_mesh("Helmet_Shell", HELMET, rows=helmet_rows(kit.RES * 0.7), res=kit.RES * 0.7, hole=vis.scaled(0.97), cap1=True,
                      mat=arm, coll=C, attrs=dict(seam=lambda g: seam_attr(
                          g, phis=[BACK, (D(40), tz(1.683), 9.0), (D(140), tz(1.683), 9.0), (D(315), 0, tz(1.609)), (D(225), 0, tz(1.609))],
                          ts=[(tz(1.611), BACK, D(100)), (tz(1.688), BACK, D(130))])))
    solid(shell, 0.008, bevel=0.002, seg=3)
    loft_mesh("Visor", HELMET, t0=vis.t0 - 0.040, t1=vis.t0 + 0.040, phi0=FRONT - 1.5, phi1=FRONT + 1.5, res=kit.RES * 0.6,
              keep=vis.scaled(1.03), offset=lambda g: -0.007 + 0.006 * (1.0 - np.clip(g.rho, 0, 1) ** 2), mat=M["visor"], coll=C)
    ph, tt = vis.outline(240, 0.985)
    P, N = HELMET.pn(ph, tt, -0.002)
    tube("Visor_Gasket", P, N, r=0.0038, closed=True, mat=M["rubber"], coll=C)

    # neck seal closing the gap under the helmet rim
    seal = Loft([ring(z, 0.0, cy, a, bf, bb, 2.1) for z, cy, a, bf, bb in (
        (1.484, -.008, .056, .057, .061), (1.508, -.006, .064, .069, .074), (1.532, -.004, .075, .089, .091))])
    loft_mesh("Neck_Seal", seal, mat=M["rubber"], coll=C)

    # respirator: snout with filter caps, pale jaw guards either side
    box("Respirator", (0.056, 0.046, 0.046), loc=(0, -0.115, 1.562), rot=(D(90), 0, 0), taper=(0.66, 0.70), bevel=0.007, seg=3,
        mat=gun, coll=C)
    for i, dz in enumerate((-0.011, 0.0, 0.011)):
        box("Respirator_Vent_%d" % i, (0.024, 0.004, 0.0035), loc=(0, -0.1385, 1.562 + dz), bevel=0.001, mat=dark, coll=C)
    cap = OnPlane((0.034, -0.110, 1.555), (0.66, 0.75, 0.0), (0.0, 0.0, 1.0))
    mirror(plate("Respirator_Filter", cap, 0.015, 0.015, n=2.0, offset=0.012, thick=0.014, bevel=0.002, mat=gun, coll=C))
    mirror(plate("Respirator_Filter_Core", cap, 0.008, 0.008, n=2.0, offset=0.0135, thick=0.003, mat=dark, coll=C))

    def jaw_row(z, o):
        return (tz(z), [(20.0, o), (42.0, o + 0.002), (62.0, o)])

    mirror(facet_loft("Jaw_Guard", HELMET, [jaw_row(1.536, .005), jaw_row(1.571, .007), jaw_row(1.602, .005)],
                      thick=0.006, bevel=0.0018, mat=arm, coll=C))

    # layered shell plates and ear modules
    plate("Helmet_Brow", OnLoft(HELMET, FRONT, tz(1.682)), 0.074, 0.008, n=6.0, offset=0.004, thick=0.006, mat=arm, coll=C)
    crown = OnEllipsoid(HC, (HA, HB, HCZ), n0=(0.0, 0.25, 1.0), up=(0.0, -1.0, 0.0))
    plate("Helmet_Crown", crown, 0.018, 0.082, n=5.0, offset=0.004, thick=0.005, mat=arm, coll=C)
    mirror(plate("Helmet_Side", OnLoft(HELMET, D(350), tz(1.694)), 0.040, 0.015, n=5.0, offset=0.0035, thick=0.005, mat=arm, coll=C))
    plate("Helmet_Nape", OnLoft(HELMET, BACK, tz(1.564)), 0.062, 0.020, n=5.0, offset=0.004, thick=0.006, mat=arm, coll=C)
    ear = OnPlane((HA - 0.004, HC[1] + 0.012, 1.620), (0, 1, 0), (0, 0, 1))
    mirror(plate("Ear_Module", ear, 0.027, 0.027, n=2.6, offset=0.014, thick=0.016, bevel=0.003, mat=gun, coll=C))
    mirror(plate("Ear_Cover", ear, 0.016, 0.016, n=2.0, offset=0.018, thick=0.005, bevel=0.0015, mat=arm, coll=C))
    mirror(plate("Ear_Core", ear, 0.007, 0.007, n=2.0, offset=0.0195, thick=0.003, mat=dark, coll=C))
    # comms stub on the right ear only
    flip_x(box("Ear_Comm", (0.011, 0.014, 0.044), loc=(HA + 0.008, HC[1] + 0.032, 1.602), rot=(D(18), 0, 0), bevel=0.003, mat=gun, coll=C))
    rivets("Helmet_Rivets", crown, [(sx, sy) for sx in (-0.030, 0.030) for sy in (-0.055, 0.0, 0.055)], 0.001, r=0.0026,
           mat=M["bronze"], coll=C)


# ------------------------------------------------------------------------ gloves
WRIST = np.array([0.265, -0.020, 0.895])
HAND_L = unit([0.12, -0.06, -1.0])
_hb = np.array([0.88, -0.47, 0.0])
HAND_B = unit(_hb - (_hb @ HAND_L) * HAND_L)


def build_gloves(M):
    C = "Gloves"
    k = 0.78
    palm, _ = parts.hand("Glove", WRIST, HAND_L, HAND_B, M["rubber"], C, scale=k, curl=1.1)
    back = OnLoft(palm, FRONT, 0.066 * k)
    mirror(plate("Glove_Plate", back, 0.029 * k, 0.028 * k, n=4.0, offset=0.004, thick=0.005, bevel=0.0015, seg=3, mat=M["armor"], coll=C))
    mirror(plate("Glove_Knuckle", back, 0.027 * k, 0.007 * k, n=5.0, shift=(0, 0.034 * k), offset=0.004, thick=0.005, mat=M["armor"], coll=C))
    f = unit(ARM.Pk[1] - ARM.Pk[0])
    cuff = Loft([dict(p=WRIST + l * f, a=a, b=a - 0.004, n=2.0) for l, a in ((-0.020, .027), (-0.014, .034), (0.004, .035), (0.022, .034), (0.028, .029))])
    mirror(loft_mesh("Glove_Cuff", cuff, mat=M["strap"], coll=C))


# ------------------------------------------------------------------------- boots
ANKLE = np.array([0.172, 0.022, 0.0])
_c, _s = math.cos(D(8)), math.sin(D(8))  # toe-out
BX, BY = np.array([_c, _s, 0.0]), np.array([-_s, _c, 0.0])
Y_TOE, Y_HEEL = -0.190, 0.075


def bw(x, y, z):
    return ANKLE + x * BX + y * BY + z * Z


def sole_top(y):
    return 0.022 + 0.012 * (y - Y_TOE) / (Y_HEEL - Y_TOE)


FOOT_PROFILE = (  # y', half-width, height of the upper over its base
    (-0.190, .004, .012), (-0.186, .022, .026), (-0.172, .036, .036), (-0.145, .045, .042), (-0.105, .049, .047),
    (-0.065, .047, .056), (-0.030, .045, .074), (0.000, .043, .094), (0.030, .041, .102), (0.055, .037, .096),
    (0.068, .026, .076), (0.074, .012, .050), (0.075, .004, .030))
FOOT = Loft([dict(p=bw(0, y, sole_top(y) + 0.008), a=a, bf=h, bb=0.012, n=(2.6, 7.0)) for y, a, h in FOOT_PROFILE], front=Z)
SOLE = Loft([dict(p=bw(0, y, sole_top(y) / 2), a=a + 0.006, bf=sole_top(y) / 2 + 0.004, bb=sole_top(y) / 2, n=7.0) for y, a, _ in FOOT_PROFILE], front=Z)
SHAFT = Loft([dict(p=bw(0, 0.006, z), a=a, bf=bf, bb=bb, n=2.2) for z, a, bf, bb in (
    (0.045, .041, .062, .052), (0.085, .040, .050, .048), (0.130, .042, .048, .050), (0.165, .045, .050, .054), (0.172, .041, .046, .050))],
    front=-BY)


def build_boots(M):
    C = "Boots"
    arm, strap = M["armor"], M["strap"]
    ft = lambda y: y - Y_TOE

    tread = parts.tread(Y_TOE, pitch=0.024, lug=0.0035, arch_y=-0.020, arch_w=0.036, arch_h=0.010)
    mirror(loft_mesh("Boot_Sole", SOLE, res=0.0028, disp=tread, cap0=True, cap1=True, mat=M["sole"], coll=C))
    mirror(loft_mesh("Boot_Foot", FOOT, res=0.003, cap0=True, cap1=True, mat=strap, coll=C))
    mirror(loft_mesh("Boot_Shaft", SHAFT, res=0.003, mat=strap, coll=C))

    def foot_row(y, spread, offs):
        return (ft(y), [(90.0 + k * spread, o) for k, o in zip((-1.0, -0.5, 0.0, 0.5, 1.0), offs)])

    mirror(facet_loft("Boot_Toe", FOOT, [
        foot_row(-0.180, 70, (.004, .005, .006, .005, .004)), foot_row(-0.150, 80, (.004, .006, .008, .006, .004)),
        foot_row(-0.105, 82, (.004, .006, .008, .006, .004)),
    ], thick=0.005, bevel=0.0015, mat=arm, coll=C))
    mirror(facet_loft("Boot_Instep", FOOT, [
        foot_row(-0.088, 62, (.004, .006, .008, .006, .004)), foot_row(-0.050, 60, (.004, .006, .009, .006, .004)),
        foot_row(-0.018, 50, (.004, .006, .008, .006, .004)),
    ], thick=0.005, bevel=0.0015, mat=arm, coll=C))
    sz = SHAFT.t_at_z

    def shaft_row(z, phis, o):
        return (sz(z), [(p, o) for p in phis])

    cuff_phis = (-40, 3, 46, 90, 134, 177, 220)
    mirror(facet_loft("Boot_Cuff", SHAFT, [shaft_row(0.098, cuff_phis, .005), shaft_row(0.152, cuff_phis, .005)],
                      thick=0.005, bevel=0.0015, mat=arm, coll=C))
    mirror(facet_loft("Boot_Heel", SHAFT, [shaft_row(0.052, (232, 270, 308), .005), shaft_row(0.092, (236, 270, 304), .005)],
                      thick=0.005, bevel=0.0015, mat=arm, coll=C))
    mirror(band_on("Boot_Strap", SHAFT, sz(0.116), sz(0.136), offset=0.012, thick=0.004, mat=strap, coll=C))
    mirror(box_at("Boot_Buckle", OnLoft(SHAFT, D(20), sz(0.128)), (0.020, 0.018, 0.005), sink=-0.0145, bevel=0.0015, mat=M["bronze"], coll=C))


def build(M):
    build_suit(M)
    build_armor(M)
    build_gear(M)
    build_helmet(M)
    build_gloves(M)
    build_boots(M)


# Beauty views: (rig azimuth, camera azimuth, elevation, distance, target height, lens mm, resolution)
BEAUTY = {
    "hero": (0.0, 30.0, 6.0, 5.0, 0.88, 85.0, (1200, 1600)),
    "hero_back": (180.0, 30.0, 6.0, 5.0, 0.88, 85.0, (1200, 1600)),
    "closeup": (0.0, 22.0, 3.0, 2.2, 1.47, 85.0, (1400, 1400)),
    "closeup_back": (180.0, 24.0, 8.0, 2.2, 1.45, 85.0, (1400, 1400)),
    "legs": (0.0, 26.0, 6.0, 2.7, 0.44, 85.0, (1400, 1400)),
    "elevated": (0.0, 35.0, 52.0, 4.8, 0.88, 85.0, (1200, 1400)),
    "top": (0.0, 0.0, 89.0, 4.2, 0.85, 85.0, (1200, 1200)),
    "art": (0.0, 6.0, -13.0, 2.9, 0.86, 50.0, (900, 1750)),  # the artwork's low camera
}
