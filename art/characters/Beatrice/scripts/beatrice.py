"""Beatrice, after `art/characters/Beatrice/without_hair.png` (front view, the modelling pose)
and `concept.jfif` (the original artwork, three-quarter, walking).

A named story character, an adult woman in a glossy black full-length bodysuit with mesh
inserts and heeled ankle boots, 1.68 m from the soles to the top of the head. THIS STAGE: the
body, suit, hands and boots; the head is a bald placeholder (its own objects in the `Head`
collection) for the later face-and-hair stage. Character only: no props.

This file is DATA; the figure is built by `charkit/bodysuit.py`:

  PALETTE   every colour on the model (sRGB hex), and the only place a colour lives
  GLOSS     the suit's and boots' gloss (a material property, tunable)
  VARIANTS  palette variants (table values only)
  DIMS      body and suit measures. The trunk and sleeve rows ARE her shape (a skin-tight suit
            is the body's own surface): heights, limbs, shoulders, arms and torso above the
            waist, knees and below fitted to the front view; hips, seat and thighs to the
            CONCEPT's proportions (Erik: the front views' hips and seat are too wide)
  SHAPES    shape variants (row overrides): `sheet_hips` = the front view's wide hips and thighs
  HEAD      the placeholder head (`charkit/parts.py::head`)

Faces -Y, +X is her LEFT; heights in metres from the soles.
"""
HEIGHT = 1.68
PREFIX = "beatrice"

# ---------------------------------------------------------------------- palette
PALETTE = dict(
    SUIT="#0e0e10",          # glossy black
    MESH="#141316",          # the inserts' net
    SKIN="#efcbbb",          # pale, as drawn
    BROW="#4a3a34",
    SCLERA="#e2dcd6",
    IRIS="#6c7488",          # grey-blue, as drawn
    LIP_TINT=(0.97, 0.80, 0.80),
    BOOT="#0d0d0f",
    SOLE="#0a0a0b",
    METAL="#b9b9be",         # the zip
)
GLOSS = dict(rough=0.16, coat=1.0, coat_rough=0.06, specular=0.5, piping=0.6,
             boot_rough=0.18, boot_coat=0.9, boot_coat_rough=0.06, mesh_cell=0.0032, mesh_show=0.55)

VARIANTS = dict(
    crimson=dict(palette=dict(SUIT="#4a0b12", BOOT="#1a0a0c", MESH="#22090c", SKIN="#a8775e", BROW="#2a1c16", IRIS="#4a3626")),
)

# ------------------------------------------------------------------- dimensions
# The trunk, ONE half (x >= 0), from inside the boots to the neck: (landmark, z, centre x,
# centre y, half-width, half-depth front, back, superellipse exponent, level section). On the
# legs (centre x > 0) the half-width is the leg's; above the crotch the section is clamped at
# the mid-plane. Front-view half-widths from the drawing (`build.py --sheet` prints them);
# the depths are hers by design (no side view): a slender, athletic profile.
LEGS = (
    ("leg_end", 0.055, .196, .010, .026, .060, .034, 2.0, 0),
    ("instep", 0.115, .193, .020, .027, .046, .034, 2.0, 0),
    ("ankle", 0.170, .189, .028, .026, .032, .032, 2.0, 0),
    ("ankle_top", 0.240, .184, .030, .027, .032, .034, 2.0, 0),
    ("shin_low", 0.300, .181, .028, .031, .036, .040, 2.0, 0),
    ("shin", 0.360, .175, .028, .041, .040, .050, 2.0, 0),
    ("calf_low", 0.400, .169, .028, .051, .042, .058, 2.0, 0),
    ("calf", 0.440, .163, .026, .057, .043, .062, 2.0, 0),
    ("calf_top", 0.480, .157, .022, .057, .044, .058, 2.0, 0),
    ("knee_low", 0.540, .1445, .014, .0495, .048, .048, 2.0, 0),
    ("knee", 0.600, .1215, .008, .0505, .050, .046, 2.0, 0),
    ("knee_top", 0.660, .1125, .006, .0535, .053, .050, 2.0, 0),
)
# hip -> thigh, CONCEPT proportions (see CONCEPT below): the outer line runs almost straight
# from the hip to the knee, the thighs meet just under the crotch
HIPS = (
    ("thigh_low", 0.720, .1075, .004, .0575, .058, .056, 2.1, 0),
    ("thigh", 0.780, .0975, .002, .0695, .066, .066, 2.1, 0),
    ("thigh_top", 0.840, .0905, .002, .0785, .072, .076, 2.15, 0),
    ("crotch", 0.893, .0865, .004, .0835, .076, .084, 2.2, 1),
    ("hip_low", 0.930, .050, .006, .107, .084, .098, 2.3, 1),
    ("hip", 0.970, .020, .006, .121, .088, .100, 2.35, 1),
    ("seat", 1.010, .000, .004, .1255, .088, .096, 2.4, 1),
    ("belly_low", 1.060, .000, .000, .108, .082, .082, 2.4, 1),
)
TORSO = (
    ("waist", 1.140, .000, .004, .094, .072, .068, 2.4, 1),
    ("waist_top", 1.180, .000, .002, .099, .076, .072, 2.4, 1),
    ("ribs", 1.220, .000, .000, .114, .086, .082, 2.45, 1),
    ("chest_low", 1.250, .000, -.004, .124, .098, .086, 2.5, 1),
    ("chest", 1.285, .000, -.006, .130, .104, .090, 2.5, 1),
    ("chest_top", 1.320, .000, -.002, .132, .096, .092, 2.5, 1),
    ("shoulder", 1.360, .000, .006, .150, .080, .090, 2.5, 1),
    ("shoulder_top", 1.395, .000, .010, .158, .064, .078, 2.4, 1),
    ("trapezius", 1.420, .000, .012, .146, .054, .066, 2.4, 1),
    ("yoke", 1.430, .000, .012, .118, .050, .060, 2.3, 1),
    ("neck_base", 1.437, .000, .012, .088, .047, .056, 2.3, 1),
    ("neck_low", 1.444, .000, .012, .066, .045, .052, 2.2, 1),
    ("neck", 1.452, .000, .012, .050, .043, .049, 2.1, 1),
    ("neck_top", 1.462, .000, .012, .046, .042, .046, 2.0, 1),
)

# Shape variants: rows replaced by landmark name.
SHAPES = dict(
    # the front view's own hips and thighs (wider), for comparison by eye
    sheet_hips=dict(rows=(
        ("thigh_low", 0.720, .109, .004, .061, .058, .056, 2.1, 0),
        ("thigh", 0.780, .105, .002, .075, .068, .068, 2.1, 0),
        ("thigh_top", 0.840, .101, .002, .085, .075, .080, 2.15, 0),
        ("crotch", 0.893, .0955, .004, .0925, .080, .090, 2.2, 1),
        ("hip_low", 0.930, .050, .006, .131, .088, .106, 2.3, 1),
        ("hip", 0.970, .020, .006, .150, .092, .108, 2.35, 1),
        ("seat", 1.010, .000, .004, .1545, .091, .102, 2.4, 1),
        ("belly_low", 1.060, .000, .000, .134, .084, .086, 2.4, 1),
    )),
)

# The concept's proportions, measured by hand on `concept.jfif` (three-quarter view, walking,
# cropped at the crotch). Only the FAR side (her left, image right) shows hip and thigh free of
# the stride and the swinging arm, so the measures are half-widths from the body's midline (zip,
# navel, crotch) to that side's outline, in concept pixels; every ratio below is between two of
# them, so the view's foreshortening cancels. Shoulder 305 (outer deltoid, y 600), waist 155
# (y 1040), hip 278 (at the crotch, y 1500), far thigh at the crotch 268 (crotch to outline).
# The hip profile from waist (0) to crotch (1), as multiples of the waist half-width:
# concept 1.05 / 1.19 / 1.39 / 1.60 / 1.79 at 0.2 / 0.4 / 0.6 / 0.8 / 1.0 (front view 1.26 / 1.51 /
# 1.70 / 1.87 / 1.97).
CONCEPT = dict(hip_shoulder=278 / 305, hip_waist=278 / 155, thigh_hip=268 / (2 * 278))


def _rows(shape=None):
    rows = list(LEGS + HIPS + TORSO)
    if shape:
        over = {r[0]: r for r in SHAPES[shape]["rows"]}
        rows = [over.get(r[0], r) for r in rows]
    return tuple(rows)


# The sleeve, wrist -> a root ring sunk inside the shoulder: (landmark, centre x, y, z,
# half-width, half-depth). Fitted to the drawing's arms (hanging out at ~22 deg).
SLEEVE = (
    ("cuff_end", .300, -.004, 0.935, .0175, .0170),
    ("cuff_top", .284, -.002, 0.968, .0195, .0185),
    ("forearm_low", .266, .000, 1.000, .0215, .0205),
    ("forearm", .2445, .002, 1.040, .0270, .0250),
    ("forearm_top", .2245, .004, 1.080, .0335, .0300),
    ("elbow", .204, .008, 1.125, .0350, .0320),
    ("elbow_top", .189, .010, 1.165, .0315, .0305),
    ("upper", .172, .012, 1.210, .0290, .0300),
    ("biceps", .162, .014, 1.260, .0300, .0320),
    ("deltoid", .154, .014, 1.310, .0355, .0390),
    ("shoulder", .150, .014, 1.370, .0380, .0420),
    ("root", .112, .014, 1.335, .0300, .0340),
)


def dims(shape=None):
    """The dimensions table for a shape (None = the concept-fitted default)."""
    return dict(
        trunk=_rows(shape),
        feature_frame=_rows("sheet_hips"),   # the suit's design is measured on the drawing's figure
        sleeve=SLEEVE,
        elbow_z=1.165,
        armpit_z=1.245,
        crotch_z=0.893,
        hand=dict(scale=0.84, curl=0.6, girth=0.70, palm_girth=0.90, drop=0.006,
                  down=(0.10, -0.04, -1.0), back=(0.92, -0.38, 0.0),
                  palm=((-0.040, .024, .019), (-0.012, .026, .018), (0.015, .033, .016), (0.045, .039, .015),
                        (0.072, .041, .014), (0.092, .039, .012), (0.102, .033, .008))),
        # broad body forms under the suit (x0, z0, side, sx, sz, height): nothing anatomical beyond them
        forms=((0.058, 1.280, "front", 0.040, 0.036, 0.024),
               (0.056, 0.950, "back", 0.055, 0.060, 0.008)),
        garment=dict(
            cloth=0.002,
            armhole=dict(x=0.128, z=1.370, tilt=4.0),
            cuff=(0.935, 0.951),
            # the stand collar's rings (z, centre y, half-width, front, back, exponent), neckline -> under the jaw
            collar=dict(rings=((1.428, .012, .060, .054, .060, 2.3), (1.445, .012, .052, .049, .055, 2.2), (1.465, .012, .046, .044, .050, 2.1),
                               (1.485, .012, .045, .043, .049, 2.0), (1.500, .012, .045, .043, .049, 2.0)),
                        thick=0.003, top_seam=0.006),
            zip=dict(bottom_z=1.040, width=0.0075, pull_z=1.488),
            knee_pad=dict(x=0.121, z=0.606, hs=0.030, ht=0.040, dome=0.004, n=2.3),
            # seams as front-view points ("x", x, z) or by section angle ("phi", deg, z)
            seams=dict(
                body=(
                    # under the bust, from the shoulder strip's end round to the princess line
                    (("x", .121, 1.316), ("x", .117, 1.284), ("x", .108, 1.256), ("x", .090, 1.238), ("x", .070, 1.234)),
                    # princess lines, under the bust down to the waist
                    (("x", .078, 1.238), ("x", .070, 1.200), ("x", .061, 1.160), ("x", .060, 1.130)),
                    # the leotard line: outer hip down to the crotch
                    (("x", .125, 1.080), ("x", .100, 1.032), ("x", .071, .984), ("x", .040, .945), ("x", .018, .918), ("x", .004, .897)),
                    # down the outer leg, from the thigh panel's tip past the knee to the boot
                    (("x", .155, .648), ("x", .170, .600), ("x", .186, .540), ("x", .200, .470), ("x", .205, .400), ("x", .199, .300),
                     ("x", .193, .232)),
                    # the knee pad's tail: a V down the shin under the pad
                    (("x", .150, .578), ("x", .121, .505), ("x", .092, .578)),
                    # --- the back (unseen; invented): centre-back seam, the leotard line under the seat,
                    # shaped seams from the armholes to the waist
                    (("phi", -90, 0.905), ("phi", -90, 1.445)),
                    (("phi", 12, 1.080), ("phi", -20, 1.050), ("phi", -50, 0.975), ("phi", -72, 0.920), ("phi", -88, 0.897)),
                    (("phi", -22, 1.300), ("phi", -40, 1.230), ("phi", -52, 1.160), ("phi", -55, 1.110)),
                ),
                arm=(),
            ),
            mesh_panels=(
                # front of the shoulders: a slanted strip from the collar's side down to the armpit's front
                dict(name="Shoulder", pts=(("x", .048, 1.418), ("x", .058, 1.414), ("x", .082, 1.392), ("x", .103, 1.362), ("x", .119, 1.326),
                                           ("x", .114, 1.320), ("x", .096, 1.352), ("x", .074, 1.382), ("x", .054, 1.405))),
                # the sides of the waist: a crescent from under the arm down and in to the waist
                dict(name="Waist", pts=(("phi", -6, 1.250), ("x", .094, 1.241), ("x", .074, 1.203), ("x", .055, 1.156), ("x", .069, 1.160),
                                        ("x", .092, 1.198), ("phi", -6, 1.205))),
                # the outer hips: a slanted strip from the waist's side down to the hip's front
                dict(name="Hip", pts=(("phi", -4, 1.132), ("x", .086, 1.110), ("x", .088, 1.074), ("x", .099, 1.050), ("x", .112, 1.073),
                                      ("phi", -4, 1.100))),
                # the outer thighs: a long almond from the hip to above the knee, round the side
                dict(name="Thigh", pts=(("x", .170, .936), ("x", .153, .920), ("x", .138, .845), ("x", .121, .782), ("x", .115, .741),
                                        ("x", .117, .700), ("x", .130, .671), ("x", .155, .648), ("phi", -12, .680), ("phi", -24, .780),
                                        ("phi", -22, .860), ("phi", -10, .920))),
            ),
        ),
        boot=dict(
            ankle=(0.193, 0.030), toe_out=14.0,
            # the foot, toe -> heel: (y along the foot, centre height, half-width, up, down); pitched on a heel
            foot=((-0.166, .016, .008, .008, .010), (-0.160, .018, .018, .013, .014), (-0.146, .020, .025, .019, .016),
                  (-0.126, .022, .030, .023, .018), (-0.100, .025, .034, .030, .020), (-0.075, .032, .034, .040, .020),
                  (-0.050, .046, .033, .046, .022), (-0.025, .062, .031, .050, .024), (0.000, .078, .031, .046, .024),
                  (0.022, .088, .032, .038, .024), (0.040, .092, .030, .030, .024), (0.052, .094, .020, .018, .016)),
            toe_cap_y=-0.112, panel_phi=24.0, sole=0.008, sole_end_y=-0.040,
            heel=(0.036, 0.022, 0.024),                    # (y, width, depth)
            shaft_z=(0.062, 0.232), shaft_ease=0.0030,    # the shaft hugs the leg this far off it
            v_depth=0.030, v_half_angle=38.0,
        ),
    )


# ------------------------------------------------------------------------ head
# The bald placeholder: the female worker's head spec re-fitted to the drawing's head outline
# (crown 1.68, cranium 0.152 m wide at 1.60, jaw 0.11 at 1.52, neck 0.080), its neck ring deep
# inside the stand collar. No face work beyond the head function's own.
HEAD = dict(
    rings=((1.404, .012, .040, .044, .044, 2.0), (1.443, .010, .040, .042, .044, 2.0), (1.474, .010, .040, .040, .046, 2.0),
           (1.487, .006, .044, .052, .054, 2.2), (1.496, -.006, .046, .080, .062, 2.2), (1.509, -.008, .052, .084, .070, 2.2),
           (1.531, -.008, .060, .087, .077, 2.2), (1.555, -.008, .066, .087, .082, 2.3), (1.583, -.008, .071, .085, .088, 2.3),
           (1.611, -.007, .075, .084, .089, 2.3)),
    crown=(1.680, 2.2),
    eye_z=1.568, nose_z=1.531, mouth_z=1.512, chin_z=1.497,
    jaw=(-0.091, 1.495, 0.021, 1.533, 0.006), jaw_soft=0.028,
    nose_proj=0.017, eye_dx=0.030, eye_r=0.0115, eye_sink=0.0005,
    nose=dict(tip_rx=0.40, tip_rz=0.42, bridge=0.0050, tip_w=0.0058),
    lids=dict(open_w=1.25, upper=0.35, lower=0.45),
    stubble=0.0,
    brow=dict(x=0.029, dz=0.0145, sx=0.016, sz=0.0018, arch=0.0035, gain=2.2),
    features=dict(
        brow=((0.028, ("eye", 0.0145), 0.019, 0.0060, 0.0026), (0.0, ("eye", 0.012), 0.012, 0.0080, 0.0010)),
        cheekbone=((0.046, ("eye", -0.018), 0.016, 0.011, 0.0050),),
        lip_upper=((0.0, ("mouth", 0.0040), 0.015, 0.0036, 0.0040),),
        lip_lower=((0.0, ("mouth", -0.0050), 0.013, 0.0042, 0.0048),),
        mouth=((0.0, ("mouth", 0.0), 0.018, 0.0012, -0.0026), (0.019, ("mouth", 0.0), 0.004, 0.004, -0.0012)),
        alae=((0.0120, ("nose", 0.0030), 0.0050, 0.0050, 0.0055),),
        alar_crease=((0.0180, ("nose", 0.0055), 0.0026, 0.0060, -0.0015), (0.0140, ("nose", 0.0090), 0.0050, 0.0022, -0.0010)),
        chin=((0.0, ("chin", 0.008), 0.017, 0.009, 0.0050), (0.0, ("mouth", -0.012), 0.014, 0.0035, -0.0016)),
    ),
    ear=dict(z=1.536, y=-0.004, h=0.048, w=0.026, tilt=14.0, flare=30.0, sink=0.002),
)

# Beauty views: (rig azimuth, camera azimuth within the rig, elevation, distance, target height, lens mm, resolution)
BEAUTY = {
    "hero": (0.0, 28.0, 6.0, 5.0, 0.84, 85.0, (1200, 1600)),
    "hero_back": (180.0, 28.0, 6.0, 5.0, 0.84, 85.0, (1200, 1600)),
    "torso": (0.0, 18.0, 4.0, 2.5, 1.20, 85.0, (1300, 1300)),
    "torso_back": (180.0, 18.0, 6.0, 2.5, 1.12, 85.0, (1300, 1300)),
    "legs": (0.0, 24.0, 6.0, 2.9, 0.46, 85.0, (1400, 1400)),
    "side": (0.0, 90.0, 4.0, 5.0, 0.84, 85.0, (1000, 1600)),
    "head": (0.0, 22.0, 3.0, 1.30, 1.56, 85.0, (1000, 1000)),
    "top": (0.0, 0.0, 89.0, 4.2, 0.84, 85.0, (1200, 1200)),
    "elevated": (0.0, 35.0, 52.0, 4.8, 0.84, 85.0, (1200, 1400)),
}

# Close-up working views, rendered on request only (`--beauty shoulder,wrist`), never in `all`
DEV_VIEWS = {
    "shoulder": (0.0, 30.0, 12.0, 1.1, 1.36, 85.0, (900, 900)),
    "wrist": (0.0, 20.0, 5.0, 0.9, 0.92, 85.0, (900, 900)),
    "boot": (0.0, 30.0, 12.0, 1.0, 0.12, 85.0, (900, 900)),
}

# The overlap bands for the sheet score: (name, z from, z to)
BANDS = (("above_waist", 1.140, 1.700), ("hip_knee", 0.600, 1.140), ("knees_down", 0.000, 0.600))


def palette(variant=None):
    p = dict(PALETTE)
    if variant:
        p.update(VARIANTS[variant].get("palette", {}))
    return p
