"""The male worker, after `art/characters/Three-view spaceship maintenance mechanic-1.png`.

A middle-aged maintenance worker in a one-piece work coverall, 1.80 m to the top of the
hair, standing with the arms down and a little out. Character only: no tools or props.

This file is DATA. The figure is built by `charkit/workwear.py` from four tables:

  PALETTE  every colour on the model (sRGB hex), and the only place a colour lives
  DIRT     strength of the grime layer (0 = clean)
  DIMS     body and garment measures; the trunk and sleeve rows are fitted to the
           sheet's silhouette (`build.py --sheet` prints the per-height mismatch)
  HEAD / HAIR   the head and the hair (`charkit/parts.py::head` / `hair`)

A different worker (the female worker, a variant) supplies her own tables with the same
keys. Faces -Y, +X is his LEFT; heights in metres from the soles.
"""
HEIGHT = 1.80

# ---------------------------------------------------------------------- palette
PALETTE = dict(
    SKIN="#946b5a",
    HAIR="#2f2722",
    COVERALL="#2f3646",     # faded navy work twill
    UNDERSHIRT="#303131",
    BOOT="#3a322c",          # worn brown-grey leather
    METAL="#7d776c",         # zips, buttons, snaps, eyelets
    SOLE="#22201e",
    LACE="#2a2420",
    STITCH="#8f7f66",        # the contrast thread of the seams
    SCLERA="#d9d2c8",
    IRIS="#5a4a3a",
    GRIME="#b39466",         # what full-strength dirt MULTIPLIES the cloth and leather by
    LIP_TINT=(1.0, 0.80, 0.78),  # lips = skin times this
)
DIRT = 1.0

# Variants change ONLY the palette and the dirt strength.
VARIANTS = dict(
    white_clean=dict(palette=dict(SKIN="#6b4632", HAIR="#1c1714", COVERALL="#e4e2dc"), dirt=0.0),
    blue=dict(palette=dict(COVERALL="#2f4f8f"), dirt=0.6),
    green=dict(palette=dict(COVERALL="#4f5d3a"), dirt=1.0),
    red=dict(palette=dict(COVERALL="#8a2a24"), dirt=0.8),
)

# ------------------------------------------------------------------- dimensions
DIMS = dict(
    # The coverall's outer surface, ONE half (x >= 0), trouser hem -> neckline. A row is
    # (landmark, height z, centre x, centre y, half-width, half-depth to the front, to the
    # back, superellipse exponent, section plane: 0 follows the limb, 1 level, or a normal). Below the crotch a row is one leg
    # (the stance is in its centre x); above, the section grows into half the pelvis and
    # is welded to its mirror at x = 0.
    trunk=(
        ("hem", 0.122, .198, .048, .078, .090, .098, 2.0, (0.0, 0.45, 1.0)),
        ("boot_top", 0.170, .192, .055, .076, .096, .097, 2.0, 0),
        ("shin_low", 0.240, .192, .058, .082, .093, .094, 2.0, 0),
        ("shin", 0.320, .176, .057, .084, .098, .098, 2.0, 0),
        ("calf", 0.400, .170, .050, .085, .102, .100, 2.0, 0),
        ("knee", 0.480, .162, .034, .079, .110, .110, 2.0, 0),
        ("knee_top", 0.560, .148, .020, .081, .114, .116, 2.0, 0),
        ("thigh_low", 0.640, .133, .010, .085, .112, .116, 2.0, 0),
        ("thigh", 0.720, .121, .006, .098, .118, .122, 2.1, 0),
        ("crotch", 0.780, .110, .002, .100, .120, .124, 2.2, 1),
        ("hip_low", 0.840, .080, .004, .136, .132, .138, 2.3, 1),
        ("hip", 0.900, .040, .010, .172, .143, .150, 2.4, 1),
        ("seat", 0.960, .010, .012, .203, .150, .152, 2.4, 1),
        ("belly_low", 1.020, .000, .004, .196, .147, .150, 2.4, 1),
        ("waist", 1.090, .000, -.006, .182, .141, .140, 2.4, 1),
        ("belly", 1.160, .000, -.002, .184, .145, .150, 2.4, 1),
        ("chest_low", 1.230, .000, .006, .192, .150, .155, 2.5, 1),
        ("chest", 1.310, .000, .012, .198, .150, .155, 2.5, 1),
        ("chest_top", 1.390, .000, .022, .195, .143, .146, 2.5, 1),
        ("shoulder", 1.450, .000, .030, .206, .125, .126, 2.5, 1),
        ("shoulder_top", 1.478, .000, .031, .192, .112, .110, 2.4, 1),
        ("yoke", 1.505, .000, .032, .160, .096, .093, 2.4, 1),
        ("collar", 1.522, .000, .030, .125, .085, .080, 2.3, 1),
        ("neck", 1.535, .000, .018, .074, .068, .068, 2.2, 1),
    ),
    # The sleeve, cuff -> a root ring sunk inside the shoulder (the armhole plane in
    # `garment` cuts it, set-in): (landmark, centre
    # x, y, z, half-width, half-depth). Its first row is the wrist.
    sleeve=(
        ("cuff_end", .388, -.040, 0.928, .041, .040),
        ("cuff_top", .378, -.032, 0.975, .044, .043),
        ("forearm", .356, -.008, 1.040, .051, .050),
        ("elbow", .328, .020, 1.110, .060, .058),
        ("upper", .282, .028, 1.200, .066, .064),
        ("biceps", .246, .028, 1.290, .066, .066),
        ("deltoid", .217, .026, 1.370, .065, .066),
        ("shoulder", .188, .025, 1.436, .058, .062),
        ("root", .150, .028, 1.440, .048, .052),   # sunk inside the body: the armhole cuts above it
    ),
    elbow_z=1.110,
    armpit_z=1.240,
    crotch_z=0.785,
    hand=dict(scale=1.04, curl=1.5, girth=0.80, palm_girth=1.0, drop=0.010,
              down=(0.20, -0.10, -1.0), back=(0.80, -0.60, 0.0),
              # bare palm sections, wrist -> knuckles: (distance down the hand, half-width, half-thickness)
              palm=((-0.040, .026, .021), (-0.012, .028, .020), (0.015, .036, .018), (0.045, .042, .0165),
                    (0.072, .044, .015), (0.092, .042, .013), (0.102, .035, .008))),
    boot=dict(
        ankle=(0.207, 0.058), toe_out=9.0, sole=(0.030, 0.038), welt=0.006,
        # foot, toe -> heel: (y along the foot from the ankle, half-width, height of the upper)
        profile=((-0.245, .004, .014), (-0.240, .024, .032), (-0.225, .040, .050), (-0.200, .048, .058), (-0.160, .051, .066),
                 (-0.120, .050, .076), (-0.080, .048, .094), (-0.040, .046, .114), (0.000, .046, .125), (0.035, .044, .128),
                 (0.060, .040, .118), (0.078, .030, .095), (0.086, .016, .060), (0.088, .004, .030)),
        toe_cap_y=-0.170, vamp_y=-0.120, lace_top_y=-0.030, lace_rows=4, shaft_y=-0.004, lace_half=0.014,
        # ankle shaft: (z, half-width, front, back)
        shaft=((0.040, .050, .058, .060), (0.090, .052, .056, .060), (0.130, .053, .056, .058), (0.165, .055, .058, .060), (0.178, .058, .062, .064)),
    ),
    undershirt=dict(rings=((1.36, .022, .150, .120, .125, 2.4), (1.43, .026, .120, .100, .100, 2.3), (1.48, .024, .085, .080, .078, 2.1),
                           (1.505, .020, .072, .073, .072, 2.0), (1.522, .018, .069, .070, .071, 2.0)), rib=0.014),
    garment=dict(
        cloth=0.0025,                 # twill thickness
        placket=0.016,                # stitch lines either side of the front zip
        zip_bottom_z=0.905,
        notch=(0.042, 0.095),         # the open neck: half-width at the top, depth
        waistband=(1.072, 1.114),
        elastic_ripples=90,          # gathers round the whole waist (only the back gathers)
        yoke_back_z=1.430,
        armhole=dict(x=0.176, z=1.47, tilt=3.0),  # the set-in sleeve's seam plane: shoulder point, lean in to the armpit
        cuff=(0.932, 0.975),
        # a low stand hugging the neck (`neck` = the neck's ellipse at the fold: cy, a, bf, bb, clearance)
        collar=dict(gap=24.0, stand=0.024, edge_drop=0.030, point_drop=0.032, neck=(0.010, 0.056, 0.058, 0.060, 0.005)),
        chest_pocket=dict(z=1.290, x=0.085, hs=0.055, ht=0.080, zip_dt=0.020),
        sleeve_pocket=dict(z=1.330, phi=12.0, hs=0.033, ht=0.050),  # left sleeve only
        back_pocket=dict(z=0.940, x=0.125, hs=0.060, ht=0.078),
        cargo=dict(z=0.705, phi=-14.0, hs=0.078, ht=0.088, depth=0.011),
        knee=dict(z=0.485, hs=0.088, ht=0.112),
    ),
    # grime by zone (charkit/workwear.py::Figure.dirt): the back and seat only fade, faintly
    dirt=dict(chest=0.75, chest_z=1.17, knees=0.62, shins=0.35, thighs=0.55, hems=0.6, forearms=0.8, back=0.12),
    dirt_front_y=0.02,
)

# ------------------------------------------------------------------------ head
HEAD = dict(
    # neck bottom (inside the collar) -> the cranium's widest ring: (z, centre y, half-width, front, back, exponent)
    rings=((1.440, .020, .062, .062, .062, 2.0), (1.490, .016, .058, .058, .060, 2.0), (1.530, .012, .053, .056, .058, 2.0),
           (1.546, .006, .057, .076, .062, 2.2), (1.556, .000, .060, .104, .068, 2.2), (1.572, -.004, .061, .110, .074, 2.2),
           (1.600, -.006, .064, .107, .076, 2.2), (1.630, -.006, .069, .104, .082, 2.3), (1.665, -.006, .074, .100, .094, 2.3),
           (1.700, -.005, .078, .098, .096, 2.3)),
    crown=(1.786, 2.2),
    eye_z=1.668, nose_z=1.628, mouth_z=1.596, chin_z=1.560,
    jaw=(-0.100, 1.552, 0.030, 1.604, 0.010),  # chin (y, z) -> jaw angle (y, z), step in to the neck
    nose_proj=0.027, eye_dx=0.032, eye_r=0.0120, eye_sink=0.0005,
    ear=dict(z=1.618, y=0.008, h=0.068, w=0.034, tilt=14.0, flare=40.0, sink=0.002),
)

HAIR = dict(
    # hairline height round the head: (phi deg, z); 90 = front, 0 / 180 = sides, 270 = back
    line=((90, 1.722), (115, 1.734), (140, 1.712), (162, 1.645), (172, 1.660), (180, 1.692), (205, 1.655), (240, 1.615),
          (270, 1.600), (300, 1.615), (335, 1.655), (0, 1.692), (8, 1.660), (18, 1.645), (40, 1.712), (65, 1.734)),
    thick=(0.003, 0.013), thick_z=(1.70, 1.765), crown_z=1.786, tufts=700, tuft_h=(0.0012, 0.0032), tuft_size=(0.005, 0.012), fade=0.006,
)

# Beauty views: (rig azimuth, camera azimuth, elevation, distance, target height, lens mm, resolution)
BEAUTY = {
    "hero": (0.0, 28.0, 6.0, 5.3, 0.90, 85.0, (1200, 1600)),
    "hero_back": (180.0, 28.0, 6.0, 5.3, 0.90, 85.0, (1200, 1600)),
    "head": (0.0, 22.0, 3.0, 1.35, 1.66, 85.0, (1200, 1200)),
    "head_side": (0.0, 75.0, 3.0, 1.35, 1.66, 85.0, (1200, 1200)),
    "torso": (0.0, 18.0, 4.0, 2.6, 1.25, 85.0, (1300, 1300)),
    "torso_back": (180.0, 18.0, 6.0, 2.6, 1.15, 85.0, (1300, 1300)),
    "legs": (0.0, 24.0, 6.0, 2.9, 0.45, 85.0, (1400, 1400)),
    "top": (0.0, 0.0, 89.0, 4.4, 0.90, 85.0, (1200, 1200)),
    "elevated": (0.0, 35.0, 52.0, 5.0, 0.90, 85.0, (1200, 1400)),
}


def palette(variant=None):
    p = dict(PALETTE)
    dirt = DIRT
    if variant:
        v = VARIANTS[variant]
        p.update(v.get("palette", {}))
        dirt = v.get("dirt", dirt)
    return p, dirt
