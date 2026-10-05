"""The officer of the Imperial Guard's light cavalry, after
`art/characters/french-imperial-guard/Gemini_Generated_Image_k4m4suk4m4suk4m4.jfif` (the sheet)
and `Skärmklipp.JPG` (the concept painting it was made from).

Hussar-style full dress: shako with plumes and cords, dolman (the braided jacket) with red
pointed cuffs, a fur-trimmed pelisse slung on his LEFT shoulder with its sleeves empty, a gold
cross-belt, a barrel sash, red breeches with Hungarian knots, Hessian boots with spurs, white
gloves. 1.80 m from the soles to the crown of the skull (hidden in the shako). Character
only: no saber, scabbard, slings or sabretache (Erik's rule).

This file is DATA. The figure is built by `assemble.py` on the kit (`charkit/workwear.Figure`
for the body and set-in sleeves, `dresswear`, `braid`, `fur`, `shako`, `parts`) from:

  PALETTE  every colour on the model (sRGB hex), the only place a colour lives
  VARIANTS palette / plume overrides only
  DIMS     body and garment measures; trunk and sleeve rows fitted to the sheet's
           silhouette (`build.py --sheet` prints the per-height mismatch)
  HEAD / HAIR   the head and hair (`charkit/parts.py::head` / `hair`), plus the moustache
  PLUMES   the plumes on the shako: one row each
  BEAUTY   the studio views

Faces -Y, +X is his LEFT; heights in metres from the soles.
"""
HEIGHT = 1.80

# ---------------------------------------------------------------------- palette
PALETTE = dict(
    SKIN="#cf9f88",
    HAIR="#5e4432",          # hair, sideburns, brows and the moustache
    SCLERA="#ddd6cc",
    IRIS="#4d5e6e",
    LIP_TINT=(1.0, 0.82, 0.80),
    DOLMAN="#151b29",        # dark navy broadcloth
    PELISSE="#151b29",
    CUFF="#b02a20",          # the dolman's pointed cuffs
    BREECHES="#a8301f",
    GOLD="#d6b062",          # lace, braid, cords, the cross-belt, boot trim
    METAL="#c9a04c",         # gilt buttons, the shako plate, the belt plate, chin scales
    FUR="#4a382a",
    FUR_TIP="#8a7058",
    SASH="#8f1a26",          # crimson cords of the barrel sash
    SASH_STRIPE="#3e0a10",
    BOOT="#191a19",          # polished black leather
    SOLE="#1c1712",
    GLOVE="#ece9e1",
    SHAKO="#151515",
    PEAK="#0d0d0d",
    COCKADE="#a8241e",
    STEEL="#8d9095",         # the spurs
    PLUME_BLACK="#141414",
    PLUME_RED="#9e2620",
)
DIRT = 0.0  # a dress uniform: no grime layer (kept for the shared driver)

# The plumes: (palette key of the colour, palette key of the tip colour or None, the tip's start
# along the plume, the stem's path base -> tip as world points, the widest radius). Front and
# back views put them side by side, black on HIS right, red on HIS left; the side view shows
# the black one leaning forward -- one 3-D arrangement fits all three.
PLUMES = (
    dict(colour="PLUME_BLACK", tip=None, tip_from=0.75, r=0.060,
         path=((-0.006, -0.050, 1.880), (-0.008, -0.090, 1.975), (-0.010, -0.142, 2.075), (-0.010, -0.150, 2.168))),
    dict(colour="PLUME_RED", tip=None, tip_from=0.75, r=0.045,
         path=((0.070, 0.004, 1.885), (0.115, 0.004, 1.960), (0.138, 0.002, 2.050), (0.140, 0.000, 2.115))),
)

# Variants change ONLY table values: palette entries and the plume rows.
VARIANTS = dict(
    green=dict(palette=dict(DOLMAN="#1f3a2c", PELISSE="#1e382a"),
               plumes=(dict(colour="PLUME_BLACK", tip="PLUME_RED", tip_from=0.78, r=0.068,
                            path=((0.020, -0.040, 1.880), (0.020, -0.090, 1.980), (0.020, -0.145, 2.080), (0.020, -0.155, 2.180))),)),
)

# ------------------------------------------------------------------- dimensions
DIMS = dict(
    # The body's outer surface, ONE half (x >= 0): breeches from inside the boot shaft up to
    # the waist, the dolman from its hem to the neckline (two meshes over one loft, split at
    # `garment.dolman_hem`). Row: (landmark, z, centre x, centre y, half-width, half-depth to the
    # front, to the back, superellipse exponent, section plane: 0 follows the limb, 1 level).
    # Below the crotch a row is one leg; above it the section grows into half the pelvis and
    # is welded to its mirror at x = 0.
    trunk=(
        ("boot_in", 0.360, .126, .030, .063, .071, .074, 2.0, 0),
        ("knee_low", 0.440, .125, .018, .064, .072, .075, 2.0, 0),
        ("knee", 0.500, .122, .004, .067, .076, .078, 2.0, 0),
        ("knee_top", 0.560, .116, -.004, .071, .080, .082, 2.0, 0),
        ("thigh_low", 0.620, .112, -.009, .075, .086, .088, 2.0, 0),
        ("thigh", 0.680, .110, -.013, .084, .094, .096, 2.1, 0),
        ("crotch", 0.735, .104, -.016, .092, .101, .104, 2.2, 1),
        ("hip_low", 0.800, .074, -.018, .126, .110, .114, 2.3, 1),
        ("hip", 0.860, .038, -.016, .160, .118, .122, 2.4, 1),
        ("seat", 0.930, .010, -.012, .186, .122, .128, 2.4, 1),
        ("belly_low", 1.000, .000, -.008, .186, .122, .122, 2.4, 1),
        ("waist", 1.080, .000, -.004, .180, .118, .114, 2.4, 1),
        ("belly", 1.150, .000, -.002, .181, .121, .114, 2.4, 1),
        ("chest_low", 1.230, .000, .000, .176, .128, .118, 2.5, 1),
        ("chest", 1.310, .000, .002, .178, .134, .120, 2.5, 1),
        ("chest_top", 1.390, .000, .006, .178, .130, .118, 2.5, 1),
        ("shoulder", 1.455, .000, .010, .184, .116, .108, 2.5, 1),
        ("shoulder_top", 1.488, .000, .012, .180, .102, .096, 2.4, 1),
        ("yoke", 1.512, .000, .012, .152, .088, .085, 2.4, 1),
        ("collar", 1.528, .000, .008, .110, .076, .072, 2.3, 1),
        ("neck", 1.540, .000, .000, .074, .066, .064, 2.2, 1),
    ),
    # The dolman sleeve, wrist -> a root ring sunk inside the shoulder (set in at the armhole):
    # (landmark, centre x, y, z, half-width, half-depth). Its first row is the wrist.
    sleeve=(
        ("cuff_end", .372, -.026, 0.955, .043, .043),
        ("cuff_top", .356, -.020, 1.010, .047, .047),
        ("forearm", .333, -.008, 1.070, .051, .052),
        ("elbow", .305, .004, 1.130, .055, .056),
        ("upper", .264, .012, 1.210, .059, .060),
        ("biceps", .230, .014, 1.290, .061, .062),
        ("deltoid", .200, .014, 1.370, .061, .063),
        ("shoulder", .178, .014, 1.445, .057, .061),
        ("root", .140, .016, 1.455, .048, .052),
    ),
    elbow_z=1.130,
    armpit_z=1.250,
    crotch_z=0.738,
    hand=dict(scale=1.0, curl=1.8, girth=1.0, palm_girth=1.0, drop=0.004, down=(0.32, -0.18, -1.0), back=(0.82, -0.55, 0.0)),
    boot=dict(
        ankle=(0.125, 0.030), toe_out=13.0, sole=0.011, heel=0.032, ball_y=-0.110, heel_front_y=0.040,
        # foot, toe -> heel: (y along the foot from the ankle, half-width, height of the upper)
        profile=((-0.222, .006, .012), (-0.214, .024, .028), (-0.198, .036, .040), (-0.172, .043, .049), (-0.135, .046, .062),
                 (-0.095, .044, .078), (-0.055, .041, .094), (-0.020, .040, .108), (0.015, .040, .110), (0.040, .039, .100),
                 (0.058, .034, .082), (0.068, .024, .060), (0.072, .012, .035), (0.073, .004, .020)),
        # the shaft, ankle -> top: (z, centre x, centre y, half-width, front, back)
        shaft=((0.050, .125, .032, .036, .042, .036), (0.100, .125, .034, .043, .048, .046), (0.160, .126, .036, .052, .058, .062),
               (0.240, .127, .034, .058, .058, .072), (0.320, .128, .030, .066, .066, .084), (0.390, .128, .026, .073, .072, .082),
               (0.470, .128, .022, .077, .078, .080)),
        top=dict(back=0.405, side=0.440, front=0.462, notch=0.405, notch_w=26.0),
        trim=0.012,
        tassel=dict(length=0.032, r_head=0.0055, r_skirt=0.0085),
        spur=dict(strap=(0.058, 0.118), r=0.0032, neck=(0.066, 0.020), rowel=(0.017, 8)),
    ),
    garment=dict(
        cloth=0.003,
        dolman_hem=1.050,              # the dolman's hem; the breeches run up under it to `breeches_top`
        breeches_top=1.120,
        armhole=dict(x=0.160, z=1.480, tilt=3.0),
        cuff=dict(top=1.040, point=0.070, point_w=70.0, point_phi=40.0, lift=0.0035, edge=0.004),   # red pointed cuff: top at the inside, point height, half-width deg
        chevrons=dict(n=2, gap=0.022, drop=0.040, half=0.060),
        collar=dict(z=(1.520, 1.590), cy=-0.012, a=0.072, bf=0.074, bb=0.070, lift=0.004, gap=0.010),
        # chest frogging: rows from z0 to z1, half-width at the bottom / top, loops, the button columns
        frogs=dict(z0=1.200, z1=1.455, n=16, w0=0.085, w1=0.118, loop=0.012, buttons=(0.0, 0.060), button_r=0.0055),
        edging=0.0030,                 # lace half-width of the hem edging, collar and seams (m)
        back=dict(curve=((0.150, 1.415), (0.128, 1.300), (0.098, 1.165)), knot=0.014, centre=(1.390, 1.160)),
        shoulder_cord=((0.040, 1.512), (0.170, 1.478)),
        belt=dict(start=(0.120, 1.470), end=(-0.182, 1.080), width=0.050, lift=0.007, badge=0.40),
        sash=dict(z=(1.075, 1.150), lift=0.010, thick=0.012, barrels=8, plate=(0.050, 0.055)),
        sash_cords=dict(hang=(-0.150, 1.080), ring=(-0.205, 1.035), tassels=((-0.110, 0.955), (-0.090, 0.950)), r=0.0035,
                        tassel=dict(length=0.062, r_head=0.008, r_skirt=0.013)),
        stripe=dict(gap=0.006),        # the breeches' double outer stripe, either side of the outer seam
        knot=dict(x=0.100, top=1.045, height=0.190, width=0.075),
        seat=dict(curve=((0.040, 1.040), (0.105, 1.010), (0.160, 0.960), (0.185, 0.905)), knot=0.010),
    ),
    pelisse=dict(
        # The pelisse LIES ON the body: its surface is derived from the dolman's (the trunk and
        # the left sleeve, blended over the shoulder and bridged where it hangs across the gap
        # between arm and side), not modelled beside it. A CARRIER loft round the trunk and the
        # left arm is only its parametrisation: each carrier point is carried in along the
        # carrier's normal onto the smooth union of the two surfaces, `clear` off them.
        # carrier rings (z, centre x, centre y, half-width, front, back), centred to his LEFT
        rings=((1.040, .060, .000, .325, .160, .150), (1.130, .058, .004, .315, .158, .152), (1.230, .054, .006, .300, .150, .156),
               (1.330, .050, .008, .286, .144, .146), (1.410, .046, .010, .265, .134, .124), (1.470, .040, .012, .232, .118, .106),
               (1.505, .026, .010, .170, .102, .094), (1.532, .010, .008, .128, .094, .088)),
        n=2.8,
        # clearance of the pelisse's OUTER surface over the trunk loft / the sleeve loft. A
        # pelisse is FUR-LINED: `thick` of cloth and lining. The dolman stands up to ~8 mm off
        # the trunk with its folds, ~4 mm off the sleeve, so 10-12 mm stay between them; near
        # its hem the dolman's own hem flares out, so the pelisse rises with it
        # (the back's folds are shallower: it lies closer there); round the neck it closes in
        # under the fur collar
        clear=dict(body=0.0300, back=0.0260, back_top=0.0180, arm=0.0340, hem=0.0340, hem_z=1.120, top=0.014, top_z=(1.480, 1.525)), thick=0.016,
        hang_z=1.330,                   # below the chest it hangs plumb from it (front and back)
        # the smooth union's fillet (m): small over the shoulder, wide where the pelisse hangs
        # across the gap between the arm and the side (below the armpit)
        blend=dict(top=0.030, low=0.120, z=(1.330, 1.140)),
        # its front edge (world x by height): from the left of the neck diagonally down the chest
        front_x=((1.100, 0.142), (1.200, 0.130), (1.320, 0.116), (1.440, 0.104), (1.535, 0.094)),
        # its other edge, across the back, from the left of the neck down to his RIGHT hip
        back_x=((1.060, -0.170), (1.200, -0.112), (1.330, -0.056), (1.450, 0.000), (1.535, 0.030)),
        # the hem round the garment: (world x, z) on the back, and on the front, outward to the
        # arm (it rises to elbow height and crosses the arm there)
        hem_back=((-0.20, 1.062), (0.10, 1.062), (0.20, 1.085), (0.27, 1.125), (0.40, 1.135)),
        hem_front=((0.10, 1.100), (0.17, 1.100), (0.24, 1.118), (0.30, 1.135), (0.40, 1.140)),
        # fur: one roll round the whole edge, radius / flattening per stretch
        fur=dict(collar=(0.031, 0.92), edge=(0.030, 0.70), hem=(0.022, 0.72),
                 # the collar is a round roll hugging the neck, round its back and left side:
                 # its centre `ring` (m from the neck's axis) at height `z`
                 ring=0.100, z=1.560),
        drape=dict(amp=0.0035, n=9, rise=0.16),     # shallow vertical folds where it hangs free, near the hem
        frogs=dict(z0=1.200, z1=1.470, n=12, x1=0.272, loop=0.008),
        # the two empty sleeves (a slung pelisse has both sleeves empty), each hanging out from
        # under the hem: LEFT from behind the elbow, down outside the forearm, its fur cuff
        # outside the hand (the back view's sleeve with chevrons and a fur cuff there, the
        # front view's dark piece with fur outside the forearm); RIGHT behind his left hip,
        # between the hip and the arm (the front view's dark piece there, the side view's
        # sleeve behind the hip with its fur cuff at the hand's level). Centre line top ->
        # bottom, half-width along `wide` / half-depth, the fur cuff (length up from the end),
        # `face` = the broad side the chevrons are on.
        sleeves=(dict(name="L", path=((0.352, 0.048, 1.150), (0.380, 0.062, 1.050), (0.392, 0.068, 0.960), (0.394, 0.070, 0.895)),
                      a=0.060, b=0.016, wide=(0.70, -0.70, 0.0), face=(0.70, 0.70, 0.0), cuff=0.070),
                 dict(name="R", path=((0.205, 0.088, 1.140), (0.226, 0.135, 1.040), (0.238, 0.166, 0.950), (0.240, 0.176, 0.880)),
                      a=0.062, b=0.016, wide=(0.70, -0.70, 0.0), face=(0.70, 0.70, 0.0), cuff=0.072)),
    ),
    shako=dict(
        rings=((1.702, -.030, .096, .106, .108), (1.746, -.024, .108, .113, .111), (1.805, -.016, .120, .121, .116),
               (1.854, -.010, .123, .126, .121), (1.894, -.006, .127, .129, .124)),
        tilt=6.0, band=(0.026, 0.002),
        peak=dict(length=0.066, droop=0.010, span=72.0, thick=0.004),
        plate=dict(dz=0.105, hs=0.034, ht=0.046, lift=0.003),
        cockade=dict(dz=0.014, r=0.016, rim=0.004),
        cords=dict(r=0.0045, front=(0.040, 0.140), back=(0.040, 0.120)),
        tassel=dict(side=-1, dz=0.050, x=0.124, y=-0.030, drop=0.185, length=0.085, r_head=0.010, r_skirt=0.017),
        chain=dict(dz=0.008, r=0.0075, flat=0.35, lift=0.010, path_z=(1.660, 1.605), path_y=(-0.050, -0.088), chin=(0.0, -0.122, 1.558)),
    ),
)

# ------------------------------------------------------------------------ head
# The male worker's head (`worker_male/scripts/worker.py::HEAD`) raised 15 mm (his eye line at
# 0.935 of 1.80 m) and set 35 mm forward: the sheet's side view carries the head ahead of the
# body's mid-plane.
HEAD = dict(
    rings=((1.455, -.004, .062, .062, .062, 2.0), (1.505, -.008, .058, .058, .060, 2.0), (1.545, -.012, .053, .056, .058, 2.0),
           (1.561, -.017, .057, .076, .062, 2.2), (1.571, -.023, .060, .104, .068, 2.2), (1.587, -.027, .061, .110, .074, 2.2),
           (1.615, -.029, .067, .107, .076, 2.2), (1.645, -.029, .073, .104, .082, 2.3), (1.680, -.029, .078, .100, .094, 2.3),
           (1.715, -.028, .081, .098, .096, 2.3)),
    crown=(1.801, 2.2),
    eye_z=1.683, nose_z=1.643, mouth_z=1.611, chin_z=1.575,
    jaw=(-0.123, 1.567, 0.007, 1.619, 0.010),
    nose_proj=0.028, eye_dx=0.032, eye_r=0.0120, eye_sink=0.0005,
    ear=dict(z=1.633, y=-0.015, h=0.066, w=0.033, tilt=14.0, flare=44.0, sink=0.002),
    stubble=0.25,
    brow=dict(gain=2.6),
    # the moustache: a full one over the upper lip, drooping a little at the ends: (x, dz from
    # the mouth, radius) from the middle out, flattened `flat` against the lip, `lift` off the skin
    moustache=dict(rows=((0.000, 0.0110, 0.0068), (0.010, 0.0108, 0.0080), (0.021, 0.0092, 0.0082), (0.032, 0.0068, 0.0068),
                         (0.041, 0.0050, 0.0048), (0.048, 0.0050, 0.0022)), flat=0.55, lift=0.0040),
)

HAIR = dict(
    # hairline height round the head: (phi deg, z); 90 = front, 0 / 180 = sides, 270 = back.
    # Short at the back and sides under the shako, sideburns down to the ear's middle.
    line=((90, 1.737), (115, 1.749), (140, 1.727), (160, 1.640), (172, 1.645), (180, 1.690), (205, 1.660), (240, 1.625),
          (270, 1.612), (300, 1.625), (335, 1.660), (0, 1.690), (8, 1.645), (20, 1.640), (40, 1.727), (65, 1.749)),
    thick=(0.003, 0.010), thick_z=(1.700, 1.760), crown_z=1.801, tufts=600, tuft_h=(0.0012, 0.0030), tuft_size=(0.005, 0.011), fade=0.006,
)

# Beauty views: (rig azimuth, camera azimuth, elevation, distance, target height, lens mm, resolution)
BEAUTY = {
    "hero": (0.0, 28.0, 6.0, 6.0, 1.02, 85.0, (1200, 1700)),
    "head": (0.0, 22.0, 3.0, 1.55, 1.70, 85.0, (1200, 1300)),
    "head_side": (0.0, 75.0, 3.0, 1.55, 1.70, 85.0, (1200, 1300)),
    "torso_front": (0.0, 12.0, 4.0, 2.4, 1.30, 85.0, (1300, 1300)),
    "torso_back": (180.0, -12.0, 6.0, 2.4, 1.28, 85.0, (1300, 1300)),
    "pelisse": (180.0, 40.0, 10.0, 3.0, 1.22, 85.0, (1300, 1400)),
    "boots": (0.0, 24.0, 8.0, 2.4, 0.30, 85.0, (1400, 1200)),
    "top": (0.0, 0.0, 89.0, 3.5, 1.00, 85.0, (1200, 1200)),
    "elevated": (0.0, 35.0, 52.0, 5.6, 1.00, 85.0, (1200, 1400)),
}


def palette(variant=None):
    """(palette, dirt) for the shared driver; the plumes are read with `plumes(variant)`."""
    p = dict(PALETTE)
    if variant:
        p.update(VARIANTS[variant].get("palette", {}))
    return p, DIRT


def plumes(variant=None):
    if variant and "plumes" in VARIANTS[variant]:
        return VARIANTS[variant]["plumes"]
    return PLUMES
