"""The female worker, after `art/characters/Spaceship maintenance mechanic turnaround sheet-2.png`.

A maintenance worker in a one-piece work coverall, 1.68 m from the soles to the top of the
hair (front view), standing with the arms down and a little out; dark brown hair pulled back
into a bun. Character only: no tools or props.

This file is DATA, with the same keys as the male worker's (`worker_male/scripts/worker.py`);
the figure is built by `charkit/workwear.py`:

  PALETTE  every colour on the model (sRGB hex), and the only place a colour lives
  DIRT     strength of the grime layer (0 = clean)
  DIMS     body and garment measures; the trunk and sleeve rows are fitted to HER sheet's
           silhouette (`build.py --sheet` prints the per-height mismatch)
  HEAD / HAIR   the head and the hair (`charkit/parts.py::head` / `hair`, `hair_bun`, `hair_strands`)

Faces -Y, +X is her LEFT; heights in metres from the soles.
"""
HEIGHT = 1.68

# ---------------------------------------------------------------------- palette
PALETTE = dict(
    SKIN="#b88a72",
    HAIR="#3a2a20",          # dark brown
    COVERALL="#3a4150",      # faded slate-navy work twill
    UNDERSHIRT="#a9a8a4",    # the light grey undershirt at the throat
    BOOT="#3b3430",
    METAL="#7d776c",
    SOLE="#22201e",
    LACE="#2a2420",
    STITCH="#8f7f66",
    SCLERA="#dcd5cb",
    IRIS="#5b4636",
    GRIME="#b39466",
    LIP_TINT=(0.98, 0.74, 0.74),
)
DIRT = 0.9

# Variants change ONLY the palette and the dirt strength.
VARIANTS = dict(
    white=dict(palette=dict(COVERALL="#e2e0da"), dirt=0.8),
    white_clean=dict(palette=dict(SKIN="#7a4e38", HAIR="#1d1512", COVERALL="#e4e2dc"), dirt=0.0),
    blue=dict(palette=dict(COVERALL="#2f4f8f"), dirt=0.6),
    green=dict(palette=dict(COVERALL="#4f5d3a"), dirt=0.9),
    red=dict(palette=dict(COVERALL="#8a2a24"), dirt=0.7),
)

# ------------------------------------------------------------------- dimensions
DIMS = dict(
    # The coverall's outer surface, ONE half (x >= 0), trouser hem -> neckline: (landmark,
    # height z, centre x, centre y, half-width, half-depth to the front, to the back,
    # superellipse exponent, section plane). Hips wider than the shoulders, a drawn-in waist,
    # the bust in the chest rows' front depth.
    trunk=(
        ("hem", 0.112, .176, .050, .073, .088, .092, 2.0, (0.0, 0.45, 1.0)),
        ("boot_top", 0.160, .175, .054, .074, .090, .090, 2.0, 0),
        ("shin_low", 0.220, .168, .056, .077, .088, .088, 2.0, 0),
        ("shin", 0.300, .162, .055, .080, .086, .086, 2.0, 0),
        ("calf", 0.380, .152, .040, .081, .092, .096, 2.0, 0),
        ("knee", 0.460, .137, .018, .075, .098, .098, 2.0, 0),
        ("knee_top", 0.540, .133, .010, .082, .098, .098, 2.0, 0),
        ("thigh_low", 0.620, .128, .003, .088, .104, .106, 2.0, 0),
        ("thigh", 0.700, .120, .001, .097, .113, .115, 2.1, 0),
        ("crotch", 0.760, .114, .002, .110, .118, .120, 2.2, 1),
        ("hip_low", 0.820, .075, .017, .147, .133, .135, 2.3, 1),
        ("hip", 0.880, .030, .016, .182, .144, .146, 2.4, 1),
        ("seat", 0.940, .008, .004, .192, .146, .146, 2.4, 1),
        ("belly_low", 1.000, .000, -.018, .180, .132, .131, 2.4, 1),
        ("waist", 1.070, .000, -.030, .158, .118, .113, 2.4, 1),
        ("belly", 1.120, .000, -.025, .162, .125, .133, 2.4, 1),
        ("chest_low", 1.180, .000, -.016, .168, .142, .135, 2.5, 1),
        ("chest", 1.230, .000, -.012, .174, .150, .145, 2.5, 1),
        ("chest_top", 1.300, .000, .005, .180, .122, .130, 2.5, 1),
        ("shoulder", 1.355, .000, .014, .185, .094, .108, 2.5, 1),
        ("shoulder_top", 1.378, .000, .016, .172, .074, .095, 2.4, 1),
        ("yoke", 1.400, .000, .018, .122, .058, .080, 2.4, 1),
        ("collar", 1.408, .000, .018, .086, .050, .068, 2.3, 1),
        ("neck", 1.420, .000, .014, .062, .056, .060, 2.2, 1),
    ),
    # The sleeve, cuff -> a root ring sunk inside the shoulder: (landmark, centre x, y, z,
    # half-width, half-depth). Its first row is the wrist.
    sleeve=(
        ("cuff_end", .340, -.035, 0.878, .040, .038),
        ("cuff_top", .330, -.028, 0.918, .043, .041),
        ("forearm", .300, -.008, 0.985, .051, .048),
        ("elbow", .262, .012, 1.075, .058, .053),
        ("upper", .239, .020, 1.150, .054, .053),
        ("biceps", .218, .022, 1.220, .051, .052),
        ("deltoid", .194, .022, 1.290, .050, .053),
        ("shoulder", .170, .022, 1.338, .047, .051),
        ("root", .132, .024, 1.318, .038, .042),
    ),
    elbow_z=1.075,
    armpit_z=1.150,
    crotch_z=0.765,
    hand=dict(scale=0.92, curl=1.5, girth=0.74, palm_girth=0.95, drop=0.010,
              down=(0.34, -0.08, -1.0), back=(0.82, -0.56, 0.0),
              palm=((-0.040, .024, .019), (-0.012, .026, .018), (0.015, .033, .016), (0.045, .039, .015),
                    (0.072, .041, .014), (0.092, .039, .012), (0.102, .033, .008))),
    boot=dict(
        ankle=(0.186, 0.050), toe_out=12.0, sole=(0.028, 0.036), welt=0.005,
        profile=((-0.222, .004, .014), (-0.218, .022, .034), (-0.205, .037, .054), (-0.184, .045, .060), (-0.149, .047, .061),
                 (-0.112, .046, .071), (-0.074, .045, .088), (-0.037, .043, .106), (0.000, .043, .116), (0.033, .041, .119),
                 (0.056, .037, .110), (0.072, .028, .088), (0.080, .015, .056), (0.082, .004, .028)),
        toe_cap_y=-0.145, vamp_y=-0.112, lace_top_y=-0.028, lace_rows=4, shaft_y=-0.004, lace_half=0.013,
        shaft=((0.037, .046, .054, .056), (0.084, .048, .052, .056), (0.121, .049, .052, .054), (0.150, .051, .054, .056), (0.162, .054, .058, .060)),
    ),
    undershirt=dict(rings=((1.27, .000, .140, .125, .120, 2.4), (1.33, .006, .110, .092, .092, 2.3), (1.37, .010, .078, .070, .072, 2.1),
                           (1.392, .010, .064, .062, .064, 2.0), (1.405, .010, .060, .060, .062, 2.0)), rib=0.010),
    garment=dict(
        cloth=0.0025,
        placket=0.016,
        zip_bottom_z=0.860,
        notch=(0.034, 0.088),         # the open neck: half-width at the top, depth
        waistband=(1.050, 1.090),
        elastic=0.0,                  # a plain waistband (no gathers on her sheet)
        elastic_ripples=90,
        yoke_back_z=1.330,
        back_centre=True,             # a centre-back seam from the yoke to the seat
        back_darts=((0.090, 1.320), (0.068, 1.090)),   # shaped seams down the back to the waist
        hip_pocket=((0.105, 1.040), (0.175, 0.930)),   # the slanted front hip pocket openings
        belt_loops=((0.095, False), (0.150, True), (0.030, True)),
        armhole=dict(x=0.152, z=1.36, tilt=3.0),
        cuff=(0.882, 0.920),
        collar=dict(gap=22.0, stand=0.020, edge_drop=0.026, point_drop=0.030, neck=(0.002, 0.055, 0.058, 0.062, 0.005)),
        # her RIGHT breast a zipped pocket, her LEFT a patch pocket with a buttoned flap and a pen slot
        chest_pocket=dict(z=1.215, x=0.085, hs=0.050, ht=0.066, zip_dt=0.014, left="flap", right="zip", flap_h=0.026, flap_point=0.006,
                          pen_slot=0.020),
        back_pocket=dict(z=0.915, x=0.105, hs=0.062, ht=0.070, point=0.014, hem_y=0.054),  # pointed hem, stitched top
        cargo=dict(z=0.655, phi=-14.0, hs=0.066, ht=0.082, depth=0.008, snaps=False),
        knee=dict(z=0.470, hs=0.078, ht=0.110),
    ),
    # grime by zone (charkit/workwear.py::Figure.dirt): the back and seat only fade, faintly
    dirt=dict(chest=0.7, chest_z=1.12, knees=0.65, shins=0.35, thighs=0.55, hems=0.6, forearms=0.8, back=0.12),
    dirt_front_y=0.0,
)

# ------------------------------------------------------------------------ head
HEAD = dict(
    # neck bottom (inside the collar) -> the cranium's widest ring: (z, centre y, half-width, front, back, exponent)
    rings=((1.350, .012, .052, .052, .052, 2.0), (1.395, .010, .049, .049, .051, 2.0), (1.432, .012, .047, .046, .052, 2.0),
           (1.447, .006, .051, .058, .060, 2.2), (1.457, -.006, .050, .088, .068, 2.2), (1.472, -.008, .054, .094, .076, 2.2),
           (1.498, -.008, .061, .095, .083, 2.2), (1.526, -.008, .067, .093, .088, 2.3), (1.559, -.008, .072, .090, .094, 2.3),
           (1.592, -.007, .075, .088, .094, 2.3)),
    crown=(1.672, 2.2),
    eye_z=1.562, nose_z=1.524, mouth_z=1.496, chin_z=1.462,
    jaw=(-0.096, 1.456, 0.022, 1.500, 0.006), jaw_soft=0.030,  # a softer jaw than his
    nose_proj=0.020, eye_dx=0.030, eye_r=0.0118, eye_sink=0.0005,
    nose=dict(tip_rx=0.40, tip_rz=0.42, bridge=0.0052, tip_w=0.0060),
    lids=dict(open_w=1.25, upper=0.35, lower=0.45),
    stubble=0.0,
    brow=dict(x=0.029, dz=0.0150, sx=0.017, sz=0.0021, arch=0.0040, gain=2.2),
    features=dict(
        brow=((0.028, ("eye", 0.0150), 0.020, 0.0065, 0.0030), (0.0, ("eye", 0.012), 0.012, 0.0080, 0.0012)),
        cheekbone=((0.048, ("eye", -0.019), 0.017, 0.012, 0.0060),),
        lip_upper=((0.0, ("mouth", 0.0045), 0.017, 0.0042, 0.0050),),
        lip_lower=((0.0, ("mouth", -0.0060), 0.015, 0.0050, 0.0060),),
        mouth=((0.0, ("mouth", 0.0), 0.021, 0.0014, -0.0032), (0.022, ("mouth", 0.0), 0.005, 0.005, -0.0016)),
        alae=((0.0130, ("nose", 0.0035), 0.0055, 0.0055, 0.0065),),
        alar_crease=((0.0200, ("nose", 0.0060), 0.0028, 0.0065, -0.0018), (0.0150, ("nose", 0.0100), 0.0055, 0.0025, -0.0012)),
        chin=((0.0, ("chin", 0.010), 0.020, 0.011, 0.0070), (0.0, ("mouth", -0.016), 0.016, 0.004, -0.0020)),
    ),
    ear=dict(z=1.517, y=-0.004, h=0.058, w=0.030, tilt=14.0, flare=34.0, sink=0.002),
)

# Hair pulled back from the face into a bun at the back of the crown; a few loose strands
# at the temples. `line` as for the male (phi deg, z): 90 = front, 0 / 180 = sides, 270 = back.
HAIR = dict(
    line=((90, 1.622), (115, 1.618), (140, 1.596), (160, 1.560), (172, 1.548), (180, 1.565), (205, 1.540), (240, 1.505),
          (270, 1.492), (300, 1.505), (335, 1.540), (0, 1.565), (8, 1.548), (20, 1.560), (40, 1.596), (65, 1.618)),
    thick=(0.0018, 0.010), thick_z=(1.575, 1.655), crown_z=1.672, tufts=1100, tuft_h=(0.0004, 0.0011), tuft_size=(0.014, 0.030),
    fade=0.005, flow_to=(0.0, 0.110, 1.610), flow_noise=0.12,
    bun=dict(c=(0.0, 0.124, 1.598), r=(0.046, 0.044, 0.036), axis=(0.0, 0.92, 0.25), coils=7, depth=0.0035),
    # loose strands: (phi deg on the head, z of the root, length, sway out, forward)
    strands=((152, 1.592, 0.062, 0.012, -0.010), (32, 1.594, 0.068, 0.012, -0.010), (24, 1.580, 0.050, 0.009, -0.006),
             (174, 1.560, 0.050, 0.010, 0.002), (6, 1.560, 0.046, 0.010, 0.002)),
)

# Beauty views: (rig azimuth, camera azimuth, elevation, distance, target height, lens mm, resolution)
BEAUTY = {
    "hero": (0.0, 28.0, 6.0, 5.0, 0.84, 85.0, (1200, 1600)),
    "hero_back": (180.0, 28.0, 6.0, 5.0, 0.84, 85.0, (1200, 1600)),
    "head": (0.0, 22.0, 3.0, 1.30, 1.55, 85.0, (1200, 1200)),
    "head_side": (0.0, 75.0, 3.0, 1.30, 1.55, 85.0, (1200, 1200)),
    "torso": (0.0, 18.0, 4.0, 2.5, 1.17, 85.0, (1300, 1300)),
    "torso_back": (180.0, 18.0, 6.0, 2.5, 1.08, 85.0, (1300, 1300)),
    "legs": (0.0, 24.0, 6.0, 2.8, 0.42, 85.0, (1400, 1400)),
    "top": (0.0, 0.0, 89.0, 4.2, 0.84, 85.0, (1200, 1200)),
    "elevated": (0.0, 35.0, 52.0, 4.8, 0.84, 85.0, (1200, 1400)),
}


def palette(variant=None):
    p = dict(PALETTE)
    dirt = DIRT
    if variant:
        v = VARIANTS[variant]
        p.update(v.get("palette", {}))
        dirt = v.get("dirt", dirt)
    return p, dirt
