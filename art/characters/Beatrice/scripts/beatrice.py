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
  PELVIS    stage 4: knee to waist authored as its OUTLINES (charkit/profiles.edge_rows), at the owner's
            "one step slimmer" size, the waist pinched; SEAT and MIDLINE finish its surface
  SHAPES    shape variants: `slimmer` / `fuller` = a step of the pelvis either side of the default;
            `stage3` (the previous default), `stage3_slim`, `sheet_hips` = the stage-3 row construction
  HEAD      the placeholder head (`charkit/parts.py::head`)

Faces -Y, +X is her LEFT; heights in metres from the soles.
"""
import numpy as np

import profiles

HEIGHT = 1.68
PREFIX = "beatrice"

# ---------------------------------------------------------------------- palette
PALETTE = dict(
    SUIT="#0e0e10",          # glossy black
    MESH="#141316",          # the inserts' net
    MESH_THIGH="#5a5a62",    # what shows through the thigh panels' net: a dark grey (the suit's, not skin)
    SKIN="#efcbbb",          # pale, as drawn
    NAIL="#f2c9c1",          # B1j: the nails, a natural light pink a touch lighter than the skin (B1i #f1c2bd read mauve end-on)
    BROW="#4a3a34",
    SCLERA="#e2dcd6",
    IRIS="#6c7488",          # grey-blue, as drawn
    LIP_TINT=(0.97, 0.80, 0.80),
    BOOT="#0d0d0f",
    SOLE="#0a0a0b",
    METAL="#8e8e94",         # the zip (stage 3: a darker, finer silver, as drawn)
    # B1j (review_stage9 item 8): the knee pads suit-dark and GLOSSY (lighter only by their highlight and their bright piped
    # rim, as drawn); B1i greyed the base (#26262c, rim #4a4a52) and the pads read as grey smoked-glass ovals
    PAD="#141418",           # the knee pads: the suit's black, a hair lifted
    PAD_RIM="#6c6c76",       # their piped rim, bright
    # B2a, the face (sampled on `concept-face-front-v1.png`, then judged on the renders)
    BLUSH="#e89a8e",         # the cheeks' blush, the nose tip's warmth
    LIP_AMOUNT=0.28,         # B2b: how much of LIP the lips take (B2a's default 0.45 read as a pink lozenge in the toon look)
    LIP="#e2a096",           # the lips, a soft rose
    LIP_LINE="#b8706a",      # the mouth line, the nostrils
    CREASE="#c99a8e",        # the lid crease
    LASH="#2a1b19",          # the upper lash band
    LASH_LOW="#7e5048",      # the lower lash line, lighter (B2b: darker, #9c6a60 vanished in the toon look)
    SCLERA_SHADE="#b9aeb0",  # the sclera under the upper lid's shadow
    IRIS_DARK="#2c2a3c",     # the iris's top and rim
    IRIS_GLOW="#9aa6c8",     # its lit lower part
    PUPIL="#16141c",
    HAIR="#241a19",          # dark brown
    HAIR_LIGHT="#3e302d",    # its streaks
    # B2b, the toon look (`LOOKS`): its absolute colours -- highlights, rims, the hair's band and the inked lines
    SHINE="#eef1f8",         # the latex's crisp streaks
    SUIT_RIM="#59606e",      # the latex's rim, a cool grey
    SKIN_RIM="#ffe2d6",      # the skin's rim, warm
    HAIR_RING="#8c7b77",     # the hair's highlight band ("angel ring"), a grey-beige as drawn
    LINE_SKIN="#7a3d36",     # the face's and hands' line, a dark red-brown as inked
    LINE_HAIR="#120b0a",
    LINE_SUIT="#020203",
    LINE_EYE="#2a1b19",
)
# B1m (the owner, 2026-10-19: "going very glossy is fine"): B1l's "wet" look is the default; B1b-B1l's satin is VARIANTS["satin"]
GLOSS = dict(rough=0.20, coat=0.85, coat_rough=0.07, specular=0.55, piping=0.9,
             nail_rough=0.20, nail_coat=0.55, nail_coat_rough=0.07,   # B1j: the nails glossier than the skin (0.48-0.60)
             # B1j: the pads glossier than the boots (coat 1.0, specular 0.8); B1l: the coat 0.6 and specular 0.5 (with the domed
             # shape: at 1.0 / 0.8 the whole face threw back the backdrop's grey as one even sheet -- the porthole -- whatever its shape)
             pad_rough=0.15, pad_coat=0.6, pad_coat_rough=0.07, pad_specular=0.5,
             boot_rough=0.22, boot_coat=0.7, boot_coat_rough=0.10, mesh_surface=True,
             # B1k (review_stage9 item 6): the inserts' net FINE and DARK with the skin showing warm through it, as drawn (the drawing's
             # chest insert: a warm tan, about #816b6a, a fine dark hatch). B1j's (cell 3.6 mm, thread share of the default
             # 0.30 = 83 % thread, the skin dimmed to 0.42, coat 0.35) read as a grey sheet perforated with coarse light dots
             mesh_cell=0.0022, mesh_width=0.06, mesh_show=0.48, mesh_coat=0.10)

VARIANTS = dict(
    crimson=dict(palette=dict(SUIT="#4a0b12", BOOT="#1a0a0c", MESH="#22090c", MESH_THIGH="#3a161a", SKIN="#a8775e", BROW="#2a1c16", IRIS="#4a3626")),
    # B1l (review_stage10 item 3): "wet" -- the suit as glossy as the patent boots (review 10 measured B1l's satin default
    # beside the drawing's wet latex: the brightest torso highlights about half the drawing's). B1m: the owner chose it, so it
    # is GLOSS itself now; the name stays (it rebuilds the default) so B1l's commands still run
    wet=dict(gloss=dict(rough=0.20, coat=0.85, coat_rough=0.07, specular=0.55)),
    # B1m: B1b-B1l's default, kept to be rebuilt (the first review called B1a's gloss chrome-like; B1b softened it to this)
    satin=dict(gloss=dict(rough=0.30, coat=0.45, coat_rough=0.20, specular=0.45)),
)

# ------------------------------------------------------------------- dimensions
# The trunk, ONE half (x >= 0), from inside the boots to the neck: (landmark, z, centre x,
# centre y, half-width, half-depth front, back, superellipse exponent, level section). On the
# legs (centre x > 0) the half-width is the leg's; above the crotch the section is clamped at
# the mid-plane. Front-view half-widths from the drawing (`build.py --sheet` prints them);
# the depths are hers by design (no side view): a slender, athletic profile.
LEGS = (
    # (centres a little in from the front view's: the back view stands narrower; the two drawings split;
    # stage 3: ankle and boot 1 mm further out, for the slimmer boot)
    ("leg_end", 0.055, 0.1921, .020, .034, .060, .062, 2.0, 0),
    ("instep", 0.115, 0.1901, .024, .0265, .036, .040, 2.0, 0),       # stage 3: inside the boot, slim (the boot's foot is its own)
    # B1h: inside the boot (from 0.10 to its top at 0.232 the boot is this leg plus the leather): the back taken in 4.5 mm (the
    # Achilles: the back line runs on straight from the visible ankle into the heel cup, not bulging behind the ankle), the
    # front 2 mm forward (the ankle's front running into the instep); B1g: ("ankle", 0.170, 0.1861, .028, .026, .042, .040)
    ("ankle", 0.170, 0.1861, .028, .026, .044, .0355, 2.0, 0),
    ("ankle_top", 0.240, 0.1811, .0235, .027, .0385, .0385, 2.0, 0),
    ("shin_low", 0.300, 0.1782, .0265, .031, .0415, .0415, 2.0, 0),
    ("shin", 0.360, 0.1719, .0285, .041, .0495, .0495, 2.0, 0),
    ("calf_low", 0.400, 0.1666, .031, .051, .057, .057, 2.0, 0),
    # B1h (review_stage7 B8): the upper calf's outer and inner swell fuller (back view, 0.46-0.52: -5..-7 % against the back
    # drawing), its back a little rounder; the ankle and the knee rows as they were (the loft's tangents at the knee unchanged,
    # so the knee from 0.54 up does not move). B1g: calf (.056, .064, .064), calf_top (.055, .067, .067)
    ("calf", 0.440, 0.1613, .032, .0570, .064, .0650, 2.0, 0),
    ("calf_top", 0.480, 0.1560, .031, .0585, .067, .0690, 2.0, 0),
    ("knee_low", 0.540, .1445, .022, .0505, .060, .060, 2.0, 0),
    ("knee", 0.600, .1215, .002, .0505, .063, .063, 2.0, 0),
    ("knee_top", 0.660, .1125, -.006, .0535, .064, .064, 2.0, 0),
)
# B1g's leg rows where B1h changed them: the pelvis profiles take the three leg rows under the knee as their lower controls
# (`_rows`, profiles.edge_rows `below`) and keep B1g's, so the body above the knee is bit-identical
LEGS_B1G = {"ankle": ("ankle", 0.170, 0.1861, .028, .026, .042, .040, 2.0, 0),
            "calf": ("calf", 0.440, 0.1613, .032, .056, .064, .064, 2.0, 0),
            "calf_top": ("calf_top", 0.480, 0.1560, .031, .055, .067, .067, 2.0, 0)}
# hip -> thigh, CONCEPT proportions in the front view (see CONCEPT below): the outer line runs
# almost straight from the hip to the knee, the thighs meet just under the crotch. Depths from the
# SIDE view with the thigh front-to-back and the seat taken in (Erik: a moderate seat, clearly
# less than the side drawing's, clearly more than stage 1's)
HIPS = (
    ("thigh_low", 0.720, .1075, -.012, .0575, .074, .074, 2.1, 0),
    ("thigh", 0.780, .0975, -.014, .0695, .081, .081, 2.1, 0),
    ("thigh_top", 0.840, .0905, -.0135, .0785, .086, .086, 2.15, 0),
    ("crotch", 0.893, .0865, -.010, .0835, .089, .089, 2.2, 1),
    ("hip_low", 0.930, .050, -.008, .107, .095, .095, 2.25, 1),
    ("hip", 0.970, .020, -.006, .121, .096, .096, 2.25, 1),
    ("seat", 1.010, .000, -.009, .1255, .096, .094, 2.25, 1),
    ("belly_low", 1.060, .000, -.025, .108, .083, .083, 2.2, 1),
)
# waist -> neck: front-view widths; depths from the side view (bust, the lumbar curve: the back's
# deepest point at the waist, 5 cm in front of the seat). Stage 3: the drawn S-curve -- the upper back
# rounded (fullest at the shoulder blades), the lumbar hollow deeper at the waist, the chest lifted
TORSO = (
    ("waist", 1.140, .000, -.027, .090, .077, .069, 2.15, 1),   # stage 4: pinched (.094 before)
    ("waist_top", 1.180, .000, -.029, .099, .081, .075, 2.15, 1),
    ("ribs", 1.220, .000, -.028, .112, .090, .086, 2.2, 1),
    ("chest_low", 1.250, .000, -.0265, .122, .0995, .0965, 2.25, 1),
    ("chest", 1.285, .000, -.021, .128, .1035, .1030, 2.3, 1),
    ("chest_top", 1.320, .000, -.013, .132, .1010, .1020, 2.3, 1),
    ("shoulder", 1.360, .000, .000, .148, .086, .090, 2.3, 1),
    ("shoulder_top", 1.395, .000, .012, .150, .068, .072, 2.2, 1),
    ("trapezius", 1.420, .000, .015, .132, .058, .062, 2.2, 1),
    ("yoke", 1.430, .000, .015, .110, .054, .058, 2.2, 1),
    ("neck_base", 1.437, .000, .012, .088, .047, .056, 2.3, 1),
    ("neck_low", 1.444, .000, .012, .060, .044, .050, 2.2, 1),
    ("neck", 1.452, .000, .012, .044, .041, .046, 2.1, 1),
    ("neck_top", 1.462, .000, .012, .040, .039, .043, 2.0, 1),
)

# Stage 4: the PELVIS as outlines (charkit/profiles.edge_rows). From above the knee to under the waist
# the trunk is authored as its silhouettes, each a smooth function of height, and generated as level
# rows every 5 mm: the hand-set HIPS rows above left the outline as straight pieces meeting in a corner
# at the crotch (the leg rows held the outer edge upright, the hip rows ran it in a straight line to
# the waist). Widths at the owner's "one step slimmer"; the waist pinched (concept: hip / waist 1.79).
#   outer  front view, the left half's outer edge: from the pinched waist ONE swell to its widest just
#          under the crotch, easing into a near-upright outer thigh down to the knee
#   inner  the gap between the legs: a rounded arch at the crotch (z 0.887), the inner thighs opening
#          evenly below it; above it the half-section's reach past the centre line (centred from 1.07)
#   front  side view: belly, groin, the thigh's forward swell into the knee
#   back   side view: the lumbar hollow, a moderate round seat whose lower half runs on into the back
#          of the thigh (no fold, no shelf)
PELVIS = dict(
    name="pelvis", z=(0.665, 1.135), step=0.005, level_from=0.80, inner_meets_outer=1.07, interp=dict(inner=("pchip", 0.895, 0.93)),
    outer=((0.70, .1635), (0.74, .1626), (0.78, .1623), (0.81, .1623), (0.84, .1615), (0.87, .1589), (0.90, .1544),
           (0.93, .1482), (0.96, .1406), (0.99, .1318), (1.02, .1220), (1.05, .1115), (1.08, .1010), (1.11, .0935)),
    inner=((0.70, .0535), (0.71, .0510), (0.74, .0430), (0.77, .0340), (0.80, .0270), (0.83, .0210), (0.85, .0170), (0.865, .0135),
           (0.875, .0100), (0.880, .0075), (0.884, .0045), (0.887, .0000), (0.895, -.0090), (0.910, -.0280), (0.930, -.0500),
           (0.960, -.0800)),
    front=((0.70, -.0790), (0.74, -.0890), (0.77, -.0950), (0.80, -.0968), (0.83, -.0972), (0.86, -.0966), (0.89, -.0960),
           (0.92, -.0965), (0.96, -.0978), (1.00, -.0993), (1.04, -.1015), (1.08, -.1033), (1.11, -.1040)),
    back=((0.70, .0600), (0.74, .0638), (0.78, .0665), (0.81, .0680), (0.84, .0705), (0.87, .0750), (0.90, .0810),
          (0.93, .0848), (0.955, .0858), (0.98, .0845), (1.01, .0800), (1.04, .0720), (1.07, .0610), (1.10, .0485), (1.12, .0440),
          (1.13, .0428)),
    cy=((0.72, -.012), (0.80, -.014), (0.86, -.012), (0.92, -.008), (0.98, -.008), (1.04, -.018), (1.10, -.026)),
    n=((0.72, 2.10), (0.84, 2.15), (0.90, 2.15), (0.98, 2.15), (1.06, 2.15)),
)
# the seat's roundness over the profile (bodysuit.Suit.forms): two broad rounds either side of the centre
SEAT4 = (0.056, 0.945, "back", 0.054, 0.065, 0.006)
PELVIS4 = PELVIS
# B1e: the seat's two rounds broader and lower over the profile (the profile itself carries the round)
SEAT = (0.056, 0.945, "back", 0.054, 0.080, 0.0045)
# B1e, the gap between the legs: fuller upper INNER thighs (concept: thigh / hip 0.482 just under the crotch).
# A narrow slot, its half-gap about 6 mm for the first 6 cm under the crotch, opening smoothly (the inner
# outline's angle to the vertical easing 3 -> 28 -> 14 deg) into the knee's stance, which stays as it was;
# at the top the half-sections close at a rate growing evenly over 4 cm (0.87 -> 0.91: the inner edge's slope
# 0.05 -> 1.3), so the midplane union's arch is round (its tip radius k |dg/dz| / 2), the saddle under it not
# flat, and the hip's three-quarter outlines take no bend at crotch height (the section's centre swings
# sideways as fast as the inner edge turns). From 0.96 up as stage 4.
PELVIS = dict(PELVIS4, inner=((0.70, .0535), (0.71, .0510), (0.74, .0430), (0.76, .0350), (0.78, .0246), (0.80, .0150), (0.815, .0105),
                              (0.83, .0080), (0.845, .0066), (0.86, .0060), (0.87, .0056), (0.874, .00515), (0.877, .0045), (0.880, .00354),
                              (0.883, .0023), (0.886, .0008), (0.890, -.0016), (0.895, -.0054), (0.900, -.0099), (0.910, -.0214),
                              (0.920, -.0345), (0.930, -.0470), (0.940, -.0590), (0.960, -.0800)),
              # B1e step 3, the front: an S, not a cone -- from the pinched waist a quicker flare that eases off over
              # the hip (after good-body-silhouette.jfif, the owner's frontal reference: the waist-to-hip gain 0.30 /
              # 0.56 / 0.78 / 0.93 of the way at 0.2 / 0.4 / 0.6 / 0.8 of the height, here taken half the way from
              # stage 4's 0.12 / 0.36 / 0.59 / 0.79: three quarters read boxy); waist and widest point as stage 4 (the
              # owner's size)
              outer=((0.70, .1635), (0.74, .1626), (0.78, .1623), (0.81, .1623), (0.84, .1618), (0.87, .1601), (0.90, .1568),
                     (0.93, .1517), (0.96, .1449), (0.99, .1363), (1.02, .1261), (1.05, .1147), (1.08, .1029), (1.11, .0938)),
              # step 2, the side: a slight roundness of the lower belly, a soft groin (0.905), the thigh's front swelling
              # 3 mm and easing into the knee
              front=((0.70, -.0790), (0.74, -.0890), (0.77, -.0955), (0.80, -.0982), (0.83, -.0984), (0.86, -.0972), (0.88, -.0962),
                     (0.905, -.0956), (0.93, -.0963), (0.96, -.0978), (0.99, -.0999), (1.02, -.1024), (1.05, -.1043), (1.08, -.1050),
                     (1.11, -.1045)),
              # the lumbar curve (a round hollow) flowing into one round seat (fullest at 0.94, the profile carrying the
              # round, SEAT only a broad low swell over it), its under-curve running over 8 cm into a slightly full back
              # of the thigh (no notch)
              back=((0.70, .0600), (0.73, .0634), (0.76, .0666), (0.79, .0692), (0.82, .0714), (0.84, .0738), (0.86, .0772),
                    (0.88, .0814), (0.90, .0849), (0.92, .0872), (0.94, .0881), (0.96, .0877), (0.98, .0860), (1.00, .0830),
                    (1.02, .0789), (1.04, .0736), (1.06, .0671), (1.08, .0598), (1.10, .0521), (1.12, .0452), (1.13, .0430)))
# stage 4 (kept as the shape `stage4`): the centre line over the pelvis (bodysuit.Suit.midline_fillet): the V
# where the half-sections are cut at the mid-plane filled with a smooth fillet, broad in front, narrow behind
MIDLINE = dict(front=0.020, back=0.008, z=(0.887, 1.12), fade=0.03, s_max=3.0)
# B1e: the pelvis as ONE surface across the mid-plane (bodysuit.MidUnion): the half-body and its mirror image
# as a smooth union from under the crotch to the waist. `reach` = the fillet's reach w either side of the plane
# (k = 2 w sigma): under the crotch it sets the arch at the slot's top and the saddle under it (the slot is
# untouched where its half-gap is at least w, closed where it falls to w / 2); above it the fillet in the V
# where the half-sections meet, broad in front (no crease down the belly), narrower behind (a soft cleft).
# `refine`: rows 0.6 mm apart round the tip and 0.1 mm at it (z from, z to, pitch, grading): the arch and the
# nearly level underside drawn by many rows; `relax`: the arch's tip (z from, z to, x under, rounds) smoothed and
# projected back onto the union (bodysuit.MidUnion.relax)
MIDPLANE = dict(z=(0.80, 1.125), band=0.03, refine=((0.866, 0.898, 0.0006, 0.008), (0.879, 0.884, 0.0001, 0.002)),
                relax=(0.874, 0.906, 0.010, 3),
                reach=((0.800, 0.0, 0.0), (0.840, 0.0065, 0.0065), (0.895, 0.0065, 0.0065), (0.930, 0.016, 0.010),
                       (0.960, 0.020, 0.010), (1.080, 0.020, 0.010), (1.125, 0.0, 0.0)))
# B1e's pelvis, kept as the shape `b1e` (the owner chooses between stage 4, B1e and the default)
PELVIS_B1E, MIDPLANE_B1E, SEAT_B1E = PELVIS, MIDPLANE, SEAT


# B1f, the gap between the legs: the inner thighs MEET. The half-bodies touch the mid-plane TANGENTIALLY at
# `contact` (their inner edge g = 0 with zero slope) and reach past it above (they overlap there, the midplane union
# filling the valley between them: one saddle, no slot); below it the half-gap opens as a convex lens (the adductor
# mass bulging toward the centre) that meets B1e's leg at `join` in value and slope, so the leg below is B1e's. The
# union bridges g < k/4, so the reach (k = 2 w sigma) decides where light first shows: MIDPLANE's `core` caps k at the
# sections' innermost depth (k0 + c dy^2, dy from the section's centre y) up to the tip, where it must stay small (a
# near-tangent cusp: tip radius k |g'| / 2, width k), while in front of and behind that depth -- the groove where the
# thighs meet -- the table's broad reach (8-12 mm) rounds the groove (a narrow one is a pinch: a bright line in gloss).
def _gap(contact, join, top, below, step=0.005):
    """The inner profile: `below`'s controls (B1e's) up to `join`; from there to the contact the cubic
    g = a dz^2 + b dz^3 (dz = contact - z) meeting `below`'s monotone curve at `join` in value AND slope, convex all the
    way (g'' > 0: the lens); above the contact the quartic a dz^2 + c dz^3 + d dz^4 with the SAME curvature at the
    contact (C2 through it: an infinite or jumping curvature there drew a line across the inner thigh in the
    three-quarter view) reaching the first `top` control (where the half-section swings to the centre line) in value
    and slope. Returns (controls, (a, b, c, d))."""
    bz, bv = np.array([q[0] for q in below]), np.array([q[1] for q in below])
    g = float(profiles.pchip(bz, bv, [join])[0])
    dg = -float(np.diff(profiles.pchip(bz, bv, [join - 5e-4, join + 5e-4]))[0] / 1e-3)   # dg/d(dz)
    d = contact - join
    a, b_ = np.linalg.solve([[d * d, d ** 3], [2 * d, 3 * d * d]], [g, dg])
    zt, gt = top[0]
    gt1 = -(top[1][1] - top[0][1]) / (top[1][0] - top[0][0])                             # dg/d(dz) at the top control
    e = contact - zt                                                                      # negative
    c_, d_ = np.linalg.solve([[e ** 3, e ** 4], [3 * e * e, 4 * e ** 3]], [gt - a * e * e, gt1 - 2 * a * e])
    zs = np.round(np.arange(join + step, zt - 1e-9, step), 4)
    dz = contact - zs
    v = np.where(dz >= 0.0, a * dz ** 2 + b_ * dz ** 3, a * dz ** 2 + c_ * dz ** 3 + d_ * dz ** 4)
    ctrl = tuple(q for q in below if q[0] <= join + 1e-9) + tuple((float(z), float(x)) for z, x in zip(zs, v)) + tuple(top)
    return ctrl, (float(a), float(b_), float(c_), float(d_))


GAP = dict(contact=0.866, join=0.78)
GAP_TOP = ((0.920, -.0345), (0.930, -.0470), (0.940, -.0590), (0.960, -.0800))
_inner, GAP["coef"] = _gap(GAP["contact"], GAP["join"], GAP_TOP, PELVIS_B1E["inner"])
PELVIS = dict(PELVIS_B1E, inner=_inner,
              # B1f step 2, the side. The front of the thigh: ONE convex swell (the quadriceps) from the groin crease
              # (0.905) to above the knee, fullest at 0.80 (6 mm in front of B1e's), easing back into the knee
              front=((0.70, -.0800), (0.74, -.0922), (0.77, -.0996), (0.80, -.1034), (0.83, -.1034), (0.86, -.1005), (0.88, -.0982),
                     (0.905, -.0963), (0.93, -.0964), (0.96, -.0978), (0.99, -.0999), (1.02, -.1024), (1.05, -.1043), (1.08, -.1050),
                     (1.11, -.1045)),
              # the back: a lumbar hollow 2 mm deeper (1.05-1.09; the waist at 1.12 as it was), a fuller seat rising out of
              # it, fullest at 0.945, tucking under over 8 cm into a back of the thigh that is slightly full (convex), no notch
              back=((0.70, .0600), (0.73, .0633), (0.76, .0666), (0.79, .0695), (0.82, .0725), (0.84, .0752), (0.86, .0794),
                    (0.87, .0821), (0.88, .0847), (0.89, .0870), (0.90, .0888), (0.91, .0900), (0.92, .0909), (0.93, .0917),
                    (0.94, .0924), (0.95, .0929), (0.96, .0931), (0.98, .0924), (1.00, .0897), (1.02, .0844), (1.04, .0765),
                    (1.06, .0675), (1.08, .0589), (1.10, .0514), (1.12, .0452), (1.13, .0430)))


# B1f step 3, the front: the early flare. The owner's frontal reference (good-body-silhouette.jfif) gains 0.30 / 0.56 /
# 0.78 / 0.93 of its waist-to-hip width at 0.2 / 0.4 / 0.6 / 0.8 of the height from the narrowest waist to the hip line
# just above the crotch (here: the narrowest waist, 1.138, to the crotch landmark, 0.893; measured on the outline the same
# way for every shape, hipsview.flare_shares). `_flare` moves B1e's outer controls in that band `step` of the way to the
# curve with exactly those shares (waist and the hip at 0.893 fixed, nothing changed at or below 0.893).
FLARE_REF = ((0.0, 0.0), (0.2, 0.30), (0.4, 0.56), (0.6, 0.78), (0.8, 0.93), (1.0, 1.0))


def _flare(outer, step, zw=1.138, zh=0.893, w=0.0900, ref=FLARE_REF):
    oz, ov = np.array([c[0] for c in outer]), np.array([c[1] for c in outer])
    H = float(profiles.natural_cubic(oz, ov, [zh])[0])
    fz, fv = np.array([r[0] for r in ref]), np.array([r[1] for r in ref])
    out = []
    for z, v in outer:
        if zh < z < zw:
            tgt = w + float(profiles.pchip(fz, fv, [(zw - z) / (zw - zh)])[0]) * (H - w)
            v = v + step * (tgt - v)
        out.append((z, float(v)))
    return tuple(out)


FLARE_STEP = 0.5
# the outer thigh tapering a little from the hip to above the knee (B1e: a parallel column); its widest stays at 0.81-0.84
TAPER = ((0.70, -.0015), (0.74, -.0018), (0.78, -.0010), (0.81, -.0003))


def _outer(step=FLARE_STEP, taper=TAPER):
    return tuple((z, v + dict(taper).get(z, 0.0)) for z, v in _flare(PELVIS_B1E["outer"], step))


PELVIS["outer"] = _outer()
# the generated rows start lower, at the knee (0.605; B1e 0.665): their junction with the sparse leg rows was a band across the
# thigh in the matte three-quarter front (quarter outline turning 2.9 deg/cm at 0.66)
PELVIS["z"] = (0.605, 1.135)
# the seat's two rounds (bodysuit.Suit.forms): taller, a little narrower and lower than B1e's, so the two buttocks read as
# two lobes with a lower edge of their own in the back views (review A3), not one dome with the lower back
SEAT = (0.056, 0.938, "back", 0.048, 0.080, 0.0065, 0.062)
MIDPLANE = dict(MIDPLANE_B1E, refine=((0.836, 0.906, 0.0006, 0.008), (0.845, 0.853, 0.0001, 0.002)), relax=(0.838, 0.925, 0.010, 4),
                core=dict(k0=0.0032, c=12.0, z=(0.851, 0.885)),
                reach=((0.800, 0.0, 0.0), (0.830, 0.0080, 0.0080), (0.855, 0.0080, 0.0080), (0.880, 0.012, 0.0095), (0.895, 0.012, 0.0095),
                       (0.910, 0.012, 0.0090), (0.930, 0.016, 0.010), (0.960, 0.020, 0.010), (1.080, 0.020, 0.010), (1.125, 0.0, 0.0)))
# B1f's pelvis, kept as the shape `b1f`
PELVIS_B1F, MIDPLANE_B1F, SEAT_B1F = PELVIS, MIDPLANE, SEAT


# B1g, the waist (review_stage6 N1). B1f's narrowest waist was where two straight cones met -- the rib cage coming down
# and the hip leaving -- turning 11 deg/cm in a few mm: a hard line across the waist in matte, a belt in the gloss. The
# outlines from the hip to under the bust are now authored by their ANGLE to the vertical (profiles.angle_controls:
# theta(z) a smooth curve through the knots, the outline its integral), so the waist is ONE long turn spread over
# about 10 cm, its narrowest part over several cm, and the profile region runs on up to `top` (the waist, waist_top,
# ribs and chest_low rows are no longer rows of their own; their widths and depths are kept: ribs, chest, the waist's
# width, height and depths). Each entry: (z from, knots ((z, deg), ...)); below `z from` the base's own controls, and
# the first two knots are the base outline's own angle there, so it is left with its slope.
WAIST = dict(
    top=1.265,
    # step 2 (review N2), the front: from above the knee. The outer thigh tapers 3 mm toward the knee (0.72) from its widest
    # (0.81) instead of standing as a column; the hip is ONE convex round from there to about 1.06, the steepest flank (27 deg),
    # where it turns into the waist's long concave turn (<= 4 deg/cm), and on up into the rib cage. Widest, hip line (0.893),
    # narrowest waist, ribs and chest as B1f. (Step 1 kept B1f's hip below 0.96 and rounded only the waist: knots
    # (0.96, -12.60) (1.00, -21.89) (1.04, -27.75) (1.08, -17.32) (1.11, -7.78) (1.138, 1.00) (1.17, 13.06) (1.20, 22.04)
    # (1.23, 19.53) (1.26, 11.02).) The share of the waist-to-hip gain at 0.2 of the height can not exceed about 0.13 with
    # the waist turning at most 4 deg/cm: the turn itself takes the first 5-7 cm under the narrowest waist
    outer=(0.66, ((0.65, 0.44), (0.66, -0.55), (0.70, -5.69), (0.74, 0.98), (0.78, 3.89), (0.82, -1.25), (0.86, -3.65), (0.90, -4.56),
                  (0.94, -8.68), (0.98, -13.71), (1.02, -24.51), (1.06, -27.08), (1.10, -14.74), (1.138, 0.18), (1.17, 11.05), (1.20, 23.82),
                  (1.23, 22.58), (1.26, 9.71))),
    # the side view's front: the belly's crest (1.08) into the rib cage (1.22) with the turn at the waist spread as far as the
    # depths allow (at most 0.5 mm from B1f's): at the centre front the glossy reflection's edge across the abdomen is where
    # this outline's normal passes through the studio's horizon, and its sharpness goes with how fast the outline turns there
    # (B1f 2.7 deg/cm at 1.146; first round 2.75; now 2.1 at most, 1.5 at the waist)
    front=(1.04, ((1.03, -3.96), (1.04, -3.34), (1.08, 0.46), (1.11, 1.36), (1.14, -3.18), (1.17, -8.99), (1.20, -11.69), (1.23, -16.66),
                  (1.26, -1.08))),
    back=(1.04, ((1.03, -21.72), (1.04, -23.58), (1.08, -24.69), (1.11, -14.04), (1.14, -3.55), (1.17, 8.40), (1.20, 18.51), (1.23, 21.64),
                 (1.26, 23.34))),
)


def _waist(base, spec=WAIST):
    """`base` (a pelvis spec) with its outlines from each entry's z on authored by angle up to spec["top"], and the
    section centre / exponent run on through the TORSO rows the region now covers."""
    sp = dict(base, z=(base["z"][0], spec["top"]))
    for k in ("outer", "front", "back"):
        if k not in spec:
            continue
        z0, knots = spec[k]
        u0 = float(profiles.natural_cubic([c[0] for c in base[k]], [c[1] for c in base[k]], [z0])[0])
        sp[k] = tuple(c for c in base[k] if c[0] < z0 - 1e-9) + profiles.angle_controls(z0, u0, knots, spec["top"])
    covered = [r for r in TORSO if r[1] < spec["top"]]
    sp["cy"] = tuple(base["cy"]) + tuple((r[1], r[3]) for r in covered)
    sp["n"] = tuple(base["n"]) + tuple((r[1], r[7]) for r in covered)
    return sp


PELVIS = _waist(PELVIS_B1F)


# B1g step 3, the inner thigh below the lens (review N4). B1f's lens (convex, 0.85 -> 0.78) met B1e's leg in a short
# concave hinge (the outline's second derivative -8 /m at 0.74-0.76) before running straight into the knee: a faint S. A
# convex lens can not run on straight into the knee's stance (its steepest slope, 0.47, is twice the knee's, 0.24), so
# the slope eases back over the whole stretch instead: from 0.70 to 0.80 the inner outline is authored by angle, its
# concave part ONE even taper (-4 /m at most, 0.71-0.77) out of the lens's convex end; the lens above 0.80 and the leg
# below 0.70 are B1f's (first and last two knots: B1f's own angle there).
GAP_LEG = (0.70, 0.80, ((0.69, -15.84), (0.70, -14.05), (0.725, -17.15), (0.75, -21.76), (0.775, -25.53), (0.80, -22.91), (0.81, -20.53),
                        (0.82, -17.79)))


def _gap_leg(inner, spec=GAP_LEG):
    z0, z1, knots = spec
    u0 = float(profiles.pchip([c[0] for c in inner], [c[1] for c in inner], [z0])[0])
    mid = profiles.angle_controls(z0, u0, knots, z1, step=0.005)[:-1]
    return tuple(c for c in inner if c[0] < z0 - 1e-9) + mid + tuple(c for c in inner if c[0] >= z1 - 1e-9)


PELVIS["inner"] = _gap_leg(PELVIS_B1F["inner"])

# B1g step 3, the groove where the thighs meet, seen from the front (review N3, the thin bright line from the seam's V
# down to the gap): its cause is the groove's floor, a round of 2-3 mm whose faces meet the plane at |n_x| 0.3-0.5
# (0.865-0.88). A broader fillet there (front reach 15-20 mm, relax box 22-30 mm wide) rounds the floor and in the gloss
# the belly's highlight runs down into the contact as one wedge -- but it draws horizontal stripes in the matte hollow
# above the cusp and a hard edge at cusp height in the three-quarter view, so the reach stays B1f's (tried and measured:
# PROVENANCE, B1g). Review N5, the hairline at the cusp's height across the thigh in the matte three-quarter front: the
# finest rows (0.1 mm, 0.845-0.853) now grade into the 0.6 mm ones over 6 mm, not 2 (an abrupt change of row pitch at
# one height); first light, the half-gap and the mesh check unchanged.
MIDPLANE = dict(MIDPLANE_B1F, refine=((0.836, 0.906, 0.0006, 0.008), (0.845, 0.853, 0.0001, 0.006)))

# A step of the pelvis (the comparison's "one step slimmer / fuller"): the outer edge (about the inner), the seat's depth
# behind the section centre and the section's reach past the centre line scaled by k, fully from the
# thigh (`full`) to the seat and fading to nothing at the waist and above the knee (`fade`). Table values only.
PELVIS_STEP = dict(full=(0.82, 1.00), fade=(0.68, 1.11))


def pelvis(k=1.0):
    if k == 1.0:
        return PELVIS
    (f0, f1), (e0, e1) = PELVIS_STEP["full"], PELVIS_STEP["fade"]

    def w(z):
        if z < f0:
            return max(0.0, (z - e0) / (f0 - e0))
        return max(0.0, min(1.0, (e1 - z) / (e1 - f1)))

    cz, cv = [c[0] for c in PELVIS["cy"]], [c[1] for c in PELVIS["cy"]]
    cy = lambda z: float(np.interp(z, cz, cv))
    sp = dict(PELVIS)
    # the outer edge about the inner one (the leg's own width below the crotch, the centre line above)
    iz, iv = [c[0] for c in PELVIS["inner"]], [max(c[1], 0.0) for c in PELVIS["inner"]]
    ref = lambda z: float(np.interp(z, iz, iv))
    sp["outer"] = tuple((z, ref(z) + (v - ref(z)) * (1.0 + (k - 1.0) * w(z))) for z, v in PELVIS["outer"])
    sp["back"] = tuple((z, cy(z) + (v - cy(z)) * (1.0 + (k - 1.0) * w(z))) for z, v in PELVIS["back"])
    # the gap stays the gap; above the crotch the section's reach past the centre line goes with the outer edge
    sp["inner"] = tuple((z, v if v >= 0.0 else v * (1.0 + (k - 1.0) * w(z))) for z, v in PELVIS["inner"])
    return sp


# Shape variants. `pelvis` = a step of the pelvis profile (above); `rows` = the stage-3 construction
# (the HIPS rows, replaced by landmark name, the waist as it was), kept for comparison and as the suit
# design's frame (`feature_frame`: the drawing's own figure)
WAIST3 = ("waist", 1.140, .000, -.027, .094, .077, .069, 2.15, 1)
SHAPES = dict(
    slimmer=dict(pelvis=0.93, seat=(0.057, 0.935, "back", 0.048, 0.065, 0.0052)),
    fuller=dict(pelvis=1.07, seat=(0.059, 0.935, "back", 0.052, 0.065, 0.0078)),
    # stage 4's default (before B1e: the half-sections cut at the mid-plane, a fillet down the centre line)
    stage4=dict(spec=PELVIS4, seat=SEAT4, midline=MIDLINE, midplane=None),
    # B1e's default (the slot between the legs, the first side profile, the half-step front flare)
    b1e=dict(spec=PELVIS_B1E, seat=SEAT_B1E, midplane=MIDPLANE_B1E),
    # B1f's default (the leg gap that meets, the half-step front flare, the waist where two cones meet)
    b1f=dict(spec=PELVIS_B1F, seat=SEAT_B1F, midplane=MIDPLANE_B1F),
    # B1f's front flare at a three-quarter and the full step to the frontal reference (B1f's default took a half)
    flare_3q=dict(spec=dict(PELVIS_B1F, outer=_outer(0.75)), seat=SEAT_B1F, midplane=MIDPLANE_B1F),
    flare_full=dict(spec=dict(PELVIS_B1F, outer=_outer(1.0)), seat=SEAT_B1F, midplane=MIDPLANE_B1F),
    # stage 3's default (the previous default, for the before/after comparison)
    stage3=dict(rows=(WAIST3,), seat=(0.058, 0.960, "back", 0.058, 0.062, 0.009)),
    # the front view's own hips and thighs (wider), for comparison by eye
    # (the three modelling-pose drawings' widths AND the side view's full seat and thigh depth)
    sheet_hips=dict(rows=(
        ("thigh_low", 0.720, .109, -.012, .061, .078, .078, 2.1, 0),
        ("thigh", 0.780, .105, -.014, .075, .092, .092, 2.1, 0),
        ("thigh_top", 0.840, .101, -.0145, .085, .0985, .0985, 2.15, 0),
        ("crotch", 0.893, .0955, -.011, .0925, .100, .100, 2.2, 1),
        ("hip_low", 0.930, .050, -.006, .131, .102, .104, 2.25, 1),
        ("hip", 0.970, .020, -.002, .150, .102, .106, 2.25, 1),
        ("seat", 1.010, .000, -.006, .1545, .101, .100, 2.25, 1),
        ("belly_low", 1.060, .000, -.025, .134, .086, .086, 2.2, 1),
        WAIST3,
    ), seat=(0.060, 0.960, "back", 0.062, 0.066, 0.014)),
    # stage 3's "one step slimmer" (the owner's choice of size: hip widths about 7 % under stage 3's default)
    stage3_slim=dict(rows=(
        ("thigh_low", 0.720, .1065, -.012, .0545, .072, .072, 2.1, 0),
        ("thigh", 0.780, .0955, -.014, .0650, .077, .077, 2.1, 0),
        ("thigh_top", 0.840, .0875, -.0135, .0730, .082, .082, 2.15, 0),
        ("crotch", 0.893, .0830, -.010, .0775, .085, .085, 2.2, 1),
        ("hip_low", 0.930, .050, -.008, .099, .091, .091, 2.25, 1),
        ("hip", 0.970, .020, -.006, .112, .092, .092, 2.25, 1),
        ("seat", 1.010, .000, -.009, .117, .092, .090, 2.25, 1),
        ("belly_low", 1.060, .000, -.025, .102, .081, .081, 2.2, 1),
        WAIST3,
    ), seat=(0.056, 0.960, "back", 0.054, 0.058, 0.006)),
)

# The hips comparison (`--compare`, shape_vs_concept.jpg): (shape, label), slimmest first, and the
# height band it shows (waist to knee, metres)
COMPARE = (("slimmer", "one step slimmer"), ("", "default (B1g)"), ("fuller", "one step fuller"))
COMPARE_BAND = (0.56, 1.22)
# the hip pictures' "befores" (`--hips`: hips_before_after.jpg, shape_vs_silhouette.jpg, side_outlines.jpg), oldest first
HIPS_BEFORE = ("stage4", "b1e", "b1f")
NOW_LABEL = "B1g"

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
    sh = SHAPES[shape] if shape else {}
    if "rows" in sh:  # the stage-3 construction: landmark rows
        over = {r[0]: r for r in sh["rows"]}
        return tuple(over.get(r[0], r) for r in LEGS + HIPS + TORSO)
    sp = sh["spec"] if "spec" in sh else pelvis(sh.get("pelvis", 1.0))
    legs = tuple(r for r in LEGS if r[1] < sp["z"][0])
    # B1f: a profile may run on through the waist (its row stays a control, it is no longer a row of its own: the loft's
    # monotone interpolation flattened the section at that one row, a ledge under it once the hip flares right below)
    # (B1g: the region may run on past several TORSO rows; the three rows above it join the controls)
    torso = tuple(r for r in TORSO if r[1] > sp["z"][1] + 1e-9)
    above = torso[:3] if sp["z"][1] < TORSO[0][1] or sp["z"][1] > TORSO[1][1] else TORSO[:4]
    below = tuple(LEGS_B1G.get(r[0], r) for r in legs[-3:])   # B1h: the pelvis's lower controls stay B1g's
    return legs + profiles.edge_rows(sp, below=below, above=above) + torso


# The sleeve, wrist -> a root ring sunk inside the shoulder: (landmark, centre x, y, z,
# half-width, half-depth). Widths fitted to the front view's arms (hanging out at ~22 deg); depths and
# the arm's place front-to-back to the side view: it hangs from 3 cm behind the body's centre at the
# shoulder to the side of the thigh at the wrist.
SLEEVE = (
    # stage 3: refitted to the front view's arm (horizontal widths x the arm's slope): a slim wrist, the
    # forearm swelling to its belly just under the elbow, a narrowing at the elbow, the upper arm, the deltoid
    ("cuff_end", .300, .006, 0.935, .0170, .0165),
    ("cuff_top", .284, .010, 0.968, .0184, .0174),
    ("forearm_low", .2579, .0159, 1.015, .0212, .0200),
    ("forearm", .2345, .0220, 1.060, .0280, .0245),
    ("forearm_top", .2131, .0278, 1.105, .0325, .0280),
    ("elbow", .1984, .0315, 1.140, .0300, .0270),
    ("elbow_top", .1852, .0349, 1.175, .0278, .0290),
    ("upper", .172, .038, 1.210, .0283, .0305),
    ("biceps", .162, .036, 1.260, .0295, .0335),
    ("deltoid", .154, .033, 1.310, .0335, .0380),
    ("shoulder", .142, .030, 1.356, .0360, .0400),
    ("root", .104, .027, 1.350, .0310, .0350),
)
# the sleeve's depth to the BACK where it differs from the front's: the point of the elbow
SLEEVE_BACK = {"elbow": .0325, "elbow_top": .0300}


# B1i (review_stage7 B2): the bare hand as ONE surface of smooth fields (bodysuit.FittedHand), replacing parts.bare_hand
# (its capsule chains and pad ellipsoids, unioned with a C1 fillet at every node, were the ring-banding). In the hand's
# frame, metres: l down the hand from the wrist point, w towards the thumb, b out of the back of the hand.
FITTED_HAND_B1I = dict(
    # the palm, ONE sweep wrist (inside the cuff) -> knuckles: (l, w centre, b centre, half-width, depth back, depth palm,
    # exponent); the rows above l 0 run up the forearm (sheared along it); it closes round over `palm_end` (l, length).
    # B1i step 2: shorter (the knuckles 13 mm nearer the wrist), tapering from the knuckles to a slimmer wrist; step 4: the
    # rows inside the cuff round (n 2) and inside the sleeve with a margin (a squarer, wider row poked 0.4 mm through the
    # sleeve above the cuff), as deep as the wrist where the cuff's underside meets it
    palm=((-0.026, 0.000, 0.0005, 0.0145, 0.0112, 0.0112, 2.0), (-0.014, 0.000, 0.0005, 0.0152, 0.0112, 0.0112, 2.1),
          (-0.005, 0.000, 0.0004, 0.0160, 0.0099, 0.0100, 2.3), (0.010, 0.0005, 0.0002, 0.0192, 0.0096, 0.0104, 2.5),
          (0.027, 0.0000, -0.0002, 0.0238, 0.0086, 0.0101, 2.5), (0.042, -0.0005, -0.0006, 0.0268, 0.0080, 0.0092, 2.5),
          (0.052, -0.0010, -0.0010, 0.0276, 0.0076, 0.0084, 2.4)),
    palm_end=(0.061, 0.013), palm_arc=(0.004, 9.0),   # the end falls 9 mm towards the little finger, 5 mm towards the index
    # the fingers: root = the knuckle joint's centre (l, w, b); splay towards the thumb, pitch out of the back, roll (rad);
    # phalanx lengths (the last to the very tip); bends at the three joints (rad, towards the palm); half-widths at the
    # knuckle, the two finger joints and where the tip starts to round; depth ratios (back, palm)
    fingers=(
        dict(name="Index", root=(0.055, 0.0200, -0.0015), splay=0.17, len=(0.044, 0.0265, 0.023), bend=(0.18, 0.34, 0.20),
             r=(0.0076, 0.0068, 0.0060, 0.0053)),
        dict(name="Middle", root=(0.058, 0.0055, -0.0002), splay=0.07, len=(0.048, 0.0295, 0.024), bend=(0.34, 0.58, 0.30),
             r=(0.0078, 0.0070, 0.0061, 0.0054)),
        dict(name="Ring", root=(0.056, -0.0088, -0.0010), splay=-0.04, len=(0.044, 0.027, 0.0220), roll=0.10, bend=(0.45, 0.72, 0.36),
             r=(0.0072, 0.0065, 0.0057, 0.0050)),
        dict(name="Pinky", root=(0.050, -0.0220, -0.0033), splay=-0.09, len=(0.0335, 0.021, 0.0190), roll=0.25, bend=(0.52, 0.80, 0.40),
             r=(0.0064, 0.0057, 0.0050, 0.0044)),
    ),
    # the thumb: from inside the ball of the thumb, its heading and nail side given directly (hand frame)
    thumb=dict(root=(0.006, 0.0110, -0.0070), dir=(0.91, 0.32, -0.27), back=(0.0, 0.82, 0.57), len=(0.036, 0.027, 0.0255),
               bend=(0.0, 0.15, 0.22), r=(0.0098, 0.0088, 0.0080, 0.0070), depth=(0.80, 0.92), s0=-0.008),
    finger_k=0.0065, thumb_k=0.012,
    # broad forms on the palm side (centre, radii along / across / deep, axis, fillet): the ball of the thumb along its
    # metacarpal, the heel under the little finger
    pads=(((0.024, 0.0140, -0.0080), (0.022, 0.0110, 0.0090), (0.85, 0.44, -0.30), 0.012),
          ((0.030, -0.0170, -0.0050), (0.024, 0.0090, 0.0080), (1.0, -0.05, 0.0), 0.012)),
    hollow=((0.034, -0.002, -0.0175), (0.018, 0.013, 0.0065), 0.006),
    nail=dict(len=0.0125, free=0.0032, width=0.72, thumb=1.1, height=0.0002),   # the free edge 3 mm short of the tip (the tip skin rounds over it)
    res=(0.0007, 0.0011), smooth=4,
)

# B1j (review_stage9 "Hands" 1-5, item 7): B1i's hand with the fingers HANGING (a gentle graded curl, close together, a
# small fan; B1i's cascade 0.72-1.72 rad curled the ring and little fingers out sideways like a claw), the back of the hand
# narrower and rounder and ending ON the knuckles (`palm_end_shear`: the back ends sooner, the palm side later; a softer
# closing, `palm_end_p`; B1i's rounded end overhung the finger roots, the ledge), the fingers thinner, tapering into a
# half-elliptic tip (`tip_p` 2), with a soft swell and a faint back crease at each finger joint (`joint`, `crease`), the
# nails small domed oval plates set in the finger (`nail["shape"]`), the wrist 1-2 mm fuller (the cuff rolls onto it)
_FINGER_FORM = dict(tip_p=2.0, joint=(0.07, 0.0035), crease=(0.06, 0.0010, 0.0005), mcp=0.10, knuckle=0.08)
FITTED_HAND = dict(
    FITTED_HAND_B1I,
    # the wrist rounder and a little fuller (B1i 32 x 20 mm a few mm under the cuff; the cuff's lower edge rolls onto it),
    # and centred in the cuff: B1i's sat 4-5 mm towards the palm in it (the gap under the cuff's edge 9 mm on the back of
    # the wrist, none on the palm side: in the front view the cuff overhung the wrist on the outside)
    palm=((-0.026, 0.000, 0.0055, 0.0145, 0.0118, 0.0118, 2.0), (-0.014, -0.0007, 0.0057, 0.0152, 0.0122, 0.0122, 2.1),
          (-0.005, -0.0015, 0.0046, 0.0160, 0.0117, 0.0123, 2.25), (0.010, 0.0005, 0.0022, 0.0197, 0.0110, 0.0114, 2.4),
          (0.027, 0.0000, 0.0005, 0.0230, 0.0085, 0.0100, 2.3), (0.042, -0.0004, -0.0006, 0.0252, 0.0074, 0.0090, 2.25),
          (0.052, -0.0008, -0.0010, 0.0256, 0.0066, 0.0082, 2.2)),
    palm_end=(0.062, 0.016), palm_end_p=2.5, palm_end_shear=0.5, palm_arc=(0.003, 9.0),
    fingers=(
        dict(_FINGER_FORM, name="Index", root=(0.055, 0.0182, -0.0012), splay=0.06, len=(0.0425, 0.0255, 0.0222),
             bend=(0.28, 0.32, 0.14), r=(0.0072, 0.0062, 0.0054, 0.0046), r_tip=0.0067),
        dict(_FINGER_FORM, name="Middle", root=(0.058, 0.0052, -0.0002), splay=0.015, len=(0.046, 0.0283, 0.0232),
             bend=(0.34, 0.40, 0.18), r=(0.0074, 0.0064, 0.0055, 0.0047), r_tip=0.0068),
        dict(_FINGER_FORM, name="Ring", root=(0.056, -0.0080, -0.0010), splay=-0.03, roll=0.05, len=(0.0423, 0.026, 0.0212),
             bend=(0.38, 0.46, 0.20), r=(0.0069, 0.0060, 0.0052, 0.0044), r_tip=0.0064),
        dict(_FINGER_FORM, name="Pinky", root=(0.050, -0.0202, -0.0030), splay=-0.075, roll=0.12, len=(0.0322, 0.0202, 0.0183),
             bend=(0.41, 0.50, 0.22), r=(0.0061, 0.0053, 0.0046, 0.0040), r_tip=0.0058),
    ),
    # B1l (review_stage10 items 7-8): the thumb with the fingers' taper and joint form (B1j: r 9.6 -> 6.2 mm, a joint swell of
    # 4 %, bends 0 / 0.15 / 0.22 -- a plain straight tube), curled a little more and heading nearer the index (B1j dir (0.91,
    # 0.32, -0.27)): from the front the thumb and the fingers no longer a pincer with an empty V between them
    thumb=dict(FITTED_HAND_B1I["thumb"], dir=(0.94, 0.21, -0.24), bend=(0.08, 0.28, 0.32), r=(0.0094, 0.0081, 0.0069, 0.0057),
               tip_p=2.2, r_tip=0.0076, joint=(0.07, 0.0035), crease=(0.06, 0.0010, 0.0005), knuckle=0.08, mcp=0.06),
    tendons=dict(h=0.00035, sig=0.0019, start=0.008, spread=0.35, fade=0.014, short=0.006, fade_end=0.004),
    nail=dict(shape="shell", len=0.0110, free=0.0022, width=0.76, thumb=1.1, height=0.00030, p=2.6, rim=0.55, sink=0.0003),
)

# the raglan cut (bodysuit.Suit.in_torso_region): front = the start of THE line, back = the raglan seam
# outside the blade insert, both down to RAGLAN_LOW under the arm (seams.body carries all three as piping)
RAGLAN_FRONT_B1J = (("x", .060, 1.432), ("x", .086, 1.402), ("x", .110, 1.366), ("x", .126, 1.330), ("x", .128, 1.300), ("x", .1275, 1.293))
RAGLAN_BACK_B1J = (("bx", .058, 1.426), ("bx", .090, 1.409), ("bx", .106, 1.380), ("bx", .117, 1.350), ("bx", .119, 1.328), ("bx", .1175, 1.293))
# B1k (review_stage9 items 4-5): B1j's lines ran 5-6 mm outside the inserts' outer edges all the way down, beside the inserts'
# own piping (two outlines, meeting in a zigzag at the tips), and behind the arm the back line passed IN FRONT of the armpit's
# fold, so the fold's back half was the body loft's, lifted onto it up to 8 mm and torn where it went deeper (the back
# armpit's tear). Now, as drawn: each line comes from under the collar higher over the shoulder, clear of the insert's upper
# half (1-2 cm), converges on its outer edge, RUNS ALONG IT to its lower tip (("panel", name): the insert's own shared edge,
# `shared`, so the edge is one piped line) and on to the armpit; the back line lies behind the fold's deep part (section angle
# below -30 deg at 1.33-1.37), so the shoulder's own surface owns the whole fold. Each starts (and the back V's centre seam
# too) a centimetre up under the collar's flare, so its cord comes out from under the collar instead of ending short of it
RAGLAN_FRONT = (("x", .046, 1.450), ("x", .056, 1.441), ("x", .067, 1.434), ("x", .080, 1.427), ("x", .094, 1.414), ("x", .106, 1.396), ("x", .114, 1.376), ("x", .119, 1.356),
                ("panel", "Shoulder"), ("x", .1272, 1.305), ("x", .1275, 1.293))
RAGLAN_BACK_B1K = (("bx", .044, 1.444), ("bx", .052, 1.434), ("bx", .062, 1.426), ("bx", .078, 1.419), ("bx", .093, 1.405), ("bx", .1025, 1.388),
                   ("bx", .1075, 1.368), ("panel", "Blade"), ("bx", .1125, 1.312), ("bx", .1125, 1.293))
# B1l (review_stage10 item 5): B1k's back line ran 4-5 mm outside the blade insert's edge from 1.39 down to 1.35 before joining
# it (two piped lines side by side for 3 cm): now it meets the edge at a clear angle at the insert's outer middle point
RAGLAN_BACK = (("bx", .044, 1.444), ("bx", .052, 1.434), ("bx", .062, 1.426), ("bx", .078, 1.419), ("bx", .093, 1.405),
               ("panel", "Blade"), ("bx", .1125, 1.312), ("bx", .1125, 1.293))
RAGLAN_LOW = 1.293
# B1l (review_stage10 item 1, the back-armpit knot): the shoulder's own surface reaches down to CUT_LOW under the arm, below
# the union's reach (it lifts the torso down to 1.278; the armpit's saddle is at 1.300). B1k's bottom line at RAGLAN_LOW cut
# through the saddle's fillet: the fillet's back face stood off the torso as a flap above a horizontal edge, under a cord.
# Now every edge of the shoulder's own surface on the torso lies where the union IS the torso (no step, no cord needed)
CUT_LOW = 1.268
# B1l: the back's cord from the collar down the blade insert's outer edge, round INTO the armpit and on down the back past the
# rib and waist inserts to the hip -- one line, as drawn (B1k: the raglan seam ran on straight down to RAGLAN_LOW past the
# insert's tip and the back seam started beside it at 1.326: two cords crossing in an X at the armpit, and a third, the
# short seam under the arm at RAGLAN_LOW, across their feet)
BACK_LINE = (RAGLAN_BACK[:6] + (("bx", .1118, 1.320), ("bx", .1085, 1.300), ("bx", .104, 1.280), ("bx", .090, 1.230), ("bx", .074, 1.190),
             ("bx", .068, 1.160), ("bx", .076, 1.120), ("bx", .098, 1.070), ("bx", .120, 1.030), ("bx", .131, .996)))

# the seam round the upper arm at the foot of the deltoid (side view: a shallow V, lowest on the outside):
# z on the outer side, z on the inner side. The shoulder's own mesh meets the sleeve exactly here.
ARM_SEAM = (1.228, 1.266)   # B1j's join and its chevron cord (the mid-arm "garter" of review 7 B7); B1k: gone

# B1k (review_stage9 items 1-2): the shoulder AND the upper arm are ONE surface from the raglan cut down to the ELBOW
# seam (bodysuit._build_shoulder_c2): the patch's lower edge -- B1j's horizontal ring at mid upper arm -- is now the
# drawn elbow seam, where the sleeve's loft takes over. Its line round the arm, (phi deg, z of the centre line): about
# level, dipping a little behind over the point of the elbow (without_hair_back.png: a soft V at about 1.155; front and
# side views 1.16-1.165)
ELBOW_SEAM = ((0, 1.160), (90, 1.164), (180, 1.161), (-90, 1.155))
# the drawn seam over the foot of the deltoid (the three drawings agree: a V lowest on the outside of the arm, rising to
# the armpit front and back; the front view's heights are kept where the side view differs, as briefed: on the surface
# 1.287 outside, 1.316 at the arm's front, 1.345 at the armpit): (phi deg, centre-line z) round the outside of the arm,
# and the heights where it leaves the raglan seams in front and behind (it ends ON their cords)
DELTOID_SEAM_B1K = dict(front=1.334, back=1.330,
                        arm=((112, 1.322), (80, 1.309), (45, 1.293), (15, 1.282), (0, 1.279), (-15, 1.282), (-45, 1.293), (-80, 1.309),
                             (-112, 1.322)))
# B1l (review_stage10 item 1): behind, it ends ON the back's one line (BACK_LINE) where that line turns into the armpit, at the
# height the back drawing shows, from the back of the arm without B1k's last point at -112 deg in the fold (the cord climbed
# out of the fold to the insert's tip at 1.330 in a hook)
DELTOID_SEAM = dict(front=1.334, back=1.312, back_line=BACK_LINE,
                    arm=((112, 1.322), (80, 1.309), (45, 1.293), (15, 1.282), (0, 1.279), (-15, 1.282), (-45, 1.293), (-80, 1.306)))

# broad body forms under the suit (x0, z0, side, sx, sz, height): the bust and the seat's
# roundness, nothing anatomical beyond them
# B1b-B1m's bust (kept as the shape `b1m`): one Gaussian a side on the chest rows' own shelf -- one flat shelf across the chest
# in front, a cone with a high apex in the side view, 20 mm further forward than the owner's profile reference
BUST_B1M = (0.060, 1.286, "front", 0.044, 0.040, 0.035)
# B1n (the owner, 2026-10-19: "the breasts could use some more work"; the profile reference concept-upper-body-side-v2.png,
# "do not go larger"): the bust as a DESIGNED field (bodysuit.bust_form), two natural volumes in a soft valley at the zip. Every
# length in metres, front-view x (her left) and height z. Measured target (PROVENANCE B1n): v2 scaled by the collar's base and
# the front waist -- apex 131 mm in front of the body's centre at about 1.28, the lower curve round and tucking into a fold at
# about 1.24, the upper slope 118 / 100 / 79 mm at 1.32 / 1.34 / 1.36
BUST = dict(
    # each breast a dome: its apex (x, z) in the front view, half-widths inwards / outwards (blended over +-w), half-heights
    # below / above (blended over +-w), height T over the wall and fullness c (the lower curve's roundness), the bell above
    # from dz on (p, w, dz): the upper slope
    # (attempt 1: apex (0.072, 1.287), a (0.080, 0.058), b (0.050, 0.095), h (0.017, 0.7), kv 0.012 -- v2's profile within 3.6 mm,
    # but in the level sections the most forward point only 44 mm out and the valley at the zip 9 mm deep: from the front still
    # close-set, one broad mass. Attempt 2: the domes 4 mm further out and stopping at the centre line (a_in = xc), a narrower
    # union there (kv): the apex 49 mm out, the valley 15 mm; their outer side 2 mm fuller, the profile kept)
    apex=(0.076, 1.288), a=(0.072, 0.060, 0.02), b=(0.048, 0.095, 0.03), h=(0.0172, 0.7), upper=(3.0, 0.025, 0.03),
    # the wall's fillet with them: the fold under (k_low), the upper slope over (k_up), across z +-w; their union at the centre
    k=(0.005, 0.008, 1.29, 0.04), kv=0.010,
    # on the front only: full in front of y -0.05, nothing behind -0.01 (the body's centre is at about -0.02 here)
    front=(-0.05, -0.01),
    # the chest wall taken back between and above the breasts (B1m's rows stood out as a shelf: 125 mm at the centre at 1.28,
    # the breasts 151): depth by height inside the window (z0, ramp, z1, ramp), faded to nothing across by x = X. Nothing
    # under 1.185 nor over 1.405 (the collar is laid on the plain loft from 1.40) moves; under the fold
    # it eases in over 5 cm (at most 1.4 mm under 1.21): a window starting AT the fold left B1m's convex underbust as a ridge
    # over a dish (a ring in the reflections under the zip's valley)
    wall=dict(z=((1.20, -0.002), (1.22, -0.004), (1.24, -0.007), (1.26, -0.009), (1.29, -0.009), (1.32, -0.008), (1.35, -0.007),
                 (1.38, -0.005)),
              win=(1.185, 0.05, 1.405, 0.03), x=0.11),
)
# B1n step 3, the seat in profile: v2's arm and hand hang over its seat, so its back outline is hidden; the skin it shows
# reaches back to the arm's front edge, so the drawn seat is AT LEAST that deep: 8-17 mm behind B1m's back outline from 0.88 to
# 0.96 m (v1 the same), with the front matching within 3-7 mm -- clearly fuller. Moved at most half way, by the seat's own two
# rounds only (no profile, no width: the back view's outline, the hip width and the leg gap stay): their height 6.5 -> 12 mm
SEAT_B1N = SEAT[:5] + (0.012,) + SEAT[6:]
FORMS = (BUST,
         SEAT_B1N)
SHAPES["b1m"] = dict(bust=BUST_B1M, seat=SEAT)   # B1n: the before (`--shape b1m`)


def dims(shape=None):
    """The dimensions table for a shape (None = the concept-fitted default)."""
    sh = SHAPES[shape] if shape else {}
    return dict(
        trunk=_rows(shape),
        feature_frame=_rows("sheet_hips"),   # the suit's design is measured on the drawing's figure
        sleeve=SLEEVE,
        sleeve_back=SLEEVE_BACK,
        elbow_z=1.140,
        armpit_z=1.245,
        crotch_z=0.893,
        hand=dict(scale=0.78, curl=1.05, fan=0.45, knuckle=0.084, finger_len=1.20, girth=0.70, palm_girth=0.88, drop=0.012,
                  down=(0.10, -0.04, -1.0), back=(0.92, -0.38, 0.0), thumb_dir=(0.80, 0.42, -0.42),
                  palm=((-0.032, .017, .014), (-0.010, .021, .016), (0.014, .032, .016), (0.040, .037, .015),
                        (0.062, .039, .014), (0.078, .036, .012), (0.088, .028, .008), (0.096, .018, .004)),
                  # the bare hand (charkit/parts.py::bare_hand): its defaults are her slender hand (B1h and before; kept, unused)
                  bare=dict(),
                  fitted=FITTED_HAND),
        forms=(sh.get("bust", FORMS[0]), sh.get("seat", FORMS[1])) + FORMS[2:],
        midline=None if "rows" in sh else sh.get("midline"),
        midplane=None if "rows" in sh else sh.get("midplane", MIDPLANE),
        garment=dict(
            cloth=0.002,
            armhole=dict(x=0.128, z=1.370, tilt=4.0),
            # torso and arm ONE surface (bodysuit.build_shoulder): the smooth union of the two lofts, a broad
            # fillet over the shoulder (the deltoid flowing into the torso), a narrow one at the armpit,
            # meshed on its own inside `box` (left shoulder, metres), at res (final, draft)
            shoulder=dict(k=(0.004, 0.024), z=(1.270, 1.345), box=((0.050, -0.100, 1.195), (0.215, 0.100, 1.447)), res=(0.0012, 0.0024),
                          join=ARM_SEAM, arm_x=0.075,
                          # a raglan cut: the shoulder's mesh is bounded by seam lines (front: THE line from the collar
                          # to the armpit; behind: the raglan seam outside the blade insert; under the arm: a short
                          # seam joining them), so every join lies under a piping cord
                          cut=dict(front=RAGLAN_FRONT, back=RAGLAN_BACK, z_low=CUT_LOW),   # B1l: B1k z_low=RAGLAN_LOW
                          # B1k (review_stage9 items 1-2): the field continuous and C2 (bodysuit.Suit.arm_proj, smin3: B1j's arm
                          # distance stepped every 0.9 mm -- the ripples), the patch reaching down to the elbow seam (`mesh_box`
                          # the region meshed; `box` still bounds the torso's part), re-projected, its normals the field's own
                          field="c2", join_curve=ELBOW_SEAM, mesh_box=((0.030, -0.100, 1.120), (0.245, 0.100, 1.447)),
                          # B1k: the torso's part of the box from x 0.030 (B1j 0.050: the shoulder's mesh ended on the box's face
                          # beside the neck in a ragged edge, the teeth over the blade insert's upper tip in flank_back.jpg)
                          box_c2=((0.030, -0.100, 1.195), (0.215, 0.100, 1.447)),
                          # the body's own lift onto the union beyond which a point counts as buried in the arm and is left
                          # (B1j 8 mm; behind the blade insert's inner edge it reaches 6)
                          edge_max=0.010,
                          # B1l (review_stage10 item 1): the arm's distance runs on past the sleeve loft's root end, fading out of the
                          # union's reach over about 2.5 cm (bodysuit.Suit.arm_proj); B1k's cut-off was a 2-4 mm cliff in the union
                          # behind the armpit (the knot)
                          root_fade=0.025,
                          # B1l: the patch's ragged edge beyond the cut's lines moved to their nearest point (Suit.cut_target)
                          cut_nearest=True,
                          # B1l: the open edges of the body, the shoulder's own surface, the sleeve and the collar marked sharp, so the
                          # solidify rim never joins their shading (bodysuit._sharp_open_edges: the teeth along the joins)
                          sharp_open_edges=True,
                          # B1l: the shoulder's own surface's edge on the torso smoothed wider in height (0.5 mm samples; B1k (8, 5))
                          region_smooth=(14, 12),
                          # the arm's own shaping in the patch (bodysuit.Suit.arm_form): the upper arm 5 % fuller (review 7 B7:
                          # 8-10 % slimmer than drawn), from the elbow seam up and gone by the shoulder's top; a rounder deltoid;
                          # a soft fold over the elbow seam, more behind (the drawings' elbow)
                          # (first pass: swell held to 1.290, deltoid 1.6 mm at 1.305 -- the widest across the deltoids in the
                          # front view 3.4 mm a side wider at 1.30, past the drawing's)
                          arm_form=dict(swell=dict(frac=0.05, z=(1.166, 1.215, 1.272, 1.330)),
                                        deltoid=dict(h=0.0010, z=1.315, sz=0.022, p=1.0),
                                        # B1l (review_stage10 item 7): no ridge over the elbow seam -- B1k's fold (h 0.6 mm, 8 mm above the
                                        # join, mostly behind) bulged the sleeve's outline just above the seam and stepped in at it: a
                                        # ring with a shelf. Now the seam is a line and the fold is the cord's own
                                        # B1k: fold=dict(h=0.0006, dz=0.0080, sz=0.0070, base=0.15, back=1.0)
                                        )),
            armhole_seam=False,
            cuff=(0.937, 0.950), cuff_lift=0.0006, cuff_thick=0.0011,   # stage 3: a thin band, as drawn
            # B1i (review_stage7 B2): ONE soft band seated on the wrist (bodysuit.build_fitted_cuffs): its top where the band's
            # was (0.950), its outer face as proud of the sleeve as the band's was (lift + thick), a rolled bottom edge just below
            # the sleeve's end (B1h: the band from 0.937 and the sleeve's own rim below it, the double ring), the underside in to
            # the skin (no gap): band, top edge radius, bottom roll radius, how far below the sleeve's end, sink (sleeve, skin)
            cuff_fit=dict(band=0.0017, top_r=0.0007, roll=0.0013, below=0.0012, sink=(0.0006, 0.0005),
                          # B1j (review_stage9 B2 cuff): from `start` under its top edge the band eases `share` of the way in
                          # towards the wrist (`thick` proud of the skin), and its lower edge rolls in onto the skin as a
                          # quarter ellipse `ry` tall ending `lip` proud (B1i: a straight tube, its flat underside standing off
                          # the wrist: cuff 1.4x the wrist in the front view, drawn 1.15x)
                          hug=dict(start=0.002, thick=0.0016, share=0.35, ry=0.0022, lip=0.0003)),
            # the stand collar's rings (z, centre y, half-width, front, back, exponent), neckline -> just
            # under the jaw (the side view: it stands straight and hugs the neck)
            collar=dict(rings=((1.428, .012, .064, .056, .062, 2.3), (1.445, .012, .052, .049, .055, 2.2), (1.458, .012, .046, .046, .052, 2.1),
                               (1.470, .012, .044, .046, .050, 2.0), (1.482, .012, .044, .046, .050, 2.0)),
                        thick=0.003, top_seam=0.005,
                        # B1k (review_stage9 item 3, review 7 B3): B1j's rings above were a straight tube standing on the shoulders
                        # (a ledge and a shadow line at its base, a flat 3 mm lip at the top standing off the neck). Now it FLARES
                        # into the shoulders (bodysuit.CollarFlare: the tube's and the body's profiles as a C2 smooth union, fillet
                        # width side / front / back) and hugs the neck at the top (1.5-2.5 mm off the placeholder neck; the drawn
                        # collar is narrower still, 40 mm half-width, but the head's neck is 40-42 mm), its top rolled in onto the
                        # neck: no lip. The height (to 1.482) and the zip running up it as before.
                        flare=dict(rings=((1.440, .010, .0433, .0425, .0480, 2.0), (1.460, .010, .0428, .0420, .0476, 2.0),
                                          (1.476, .010, .0428, .0424, .0479, 2.0), (1.482, .010, .0430, .0428, .0482, 2.0)),
                                   top=1.482, roll=0.0018, k=(0.034, 0.016, 0.024), under=0.00008, drop=None, bend=0.003, gap=(0.006, 0.0015, 0.003), z0=1.40,
                                   start=0.004, res=0.0012),
                        flare_thick=0.002),
            # the zips, front (collar -> below the navel) and back (collar -> the small of the back): teeth on a
            # dark tape, a slider with its pull lying flat at the collar's top
            zip=dict(bottom_z=1.040, back_bottom_z=1.165, width=0.0030, tape=0.0052, pitch=0.0013, teeth=True,
                     tooth=(0.0008, 0.0007), pull_scale=0.72,   # stage 3: fine, as drawn
                     # B1g: the stop at the foot of each zip a flat plate (width, length, thickness, sunk), lying on the suit
                     stop=(0.0060, 0.0045, 0.0011, 0.0004)),
            # stage 3: layered pads on the FRONT of the knee (a pointed shield, a smaller plate over its top);
            # behind the knee only a seam
            # B1h (review_stage7 B5): ONE domed shield as drawn (bodysuit.build_knee_pads), 0.54 of the knee's width (B1g: 0.43,
            # two stacked plates whose stitch line showed dashed), its top and point where B1g's were; a piped rim (radius, centre
            # at this share of the pad's edge height). B1g: layers=(dict(x=.121, z=.612, hs=.0245, ht=.040, offset=.0016,
            # thick=.0016, dome=.0022, n=2.3, inset=.004, point=.010), dict(x=.121, z=.634, hs=.0175, ht=.0225, offset=.0031,
            # thick=.0016, dome=.0016, n=2.2, inset=.0035))
            # B1i (review_stage8 1): the dome 9 -> 3.6 mm over a 1.4 mm edge (the crest about 5 mm proud of the knee, B1h about 11),
            # and below 0.2 of its half-height under the centre the edge sinks to 0.25 mm at the point (`lower`): its lower part
            # runs into the shin, one knee line in the side view; its own lighter finish (`PAD`, `PAD_RIM`)
            knee_pad=dict(pad=dict(x=.121, z=.612, hs=.031, ht=.0425, offset=.0014, thick=.0018, dome=.0036, n=2.2, point=.010,
                                   rim=(0.0010, 0.5), res=0.0010, lower=(-0.2, 0.00025),
                                   # B1l (review_stage10 item 2): a DOME with a crest (bodysuit._DomedAnchor) -- B1i's 3.6 mm parabola
                                   # over the 62 x 95 mm pad read as a flat mirror porthole; a crowned dome, its edge rolling down
                                   # steeply under the rim (p < 1) so the highlight runs round it as a streak, a soft ridge down its
                                   # middle; the side view 7-8 mm proud (B1h 18 mm at a 9 mm dome)
                                   profile=dict(dome=0.0044, p=0.6, crest=0.0015, w=0.30, cy=0.15, sy=0.9))),
            panel_rim="seam",  # mesh panels set IN the suit, edged by the suit's own seams
            seam_style="tube", piping=(0.0009, 0.0001),  # seams are piping cords (radius, centre above the surface)
            # seams as points seen on the front view ("x", x, z), on the back view ("bx", x, z), or by
            # section angle ("phi", deg, z); x on the drawings' figure, z in metres
            seams=dict(
                body=(
                    # THE line: from the collar's side over the front of the shoulder (outside the shoulder
                    # slit), round the side of the bust, past the waist crescent, out to the hip
                    RAGLAN_FRONT + (("x", .123, 1.270), ("x", .112, 1.240), ("x", .098, 1.208), ("x", .086, 1.178), ("x", .079, 1.150),
                     ("x", .080, 1.122), ("x", .091, 1.098), ("x", .114, 1.072), ("x", .143, 1.046), ("phi", 0, 1.030)),
                    # under the bust, from the side to the zip. B1n: IN the new bust's fold, 2 mm under it, as drawn (the concept's
                    # seam traces each breast's lower curve and meets the other at the zip); B1b-B1m: (.118, 1.262) (.100, 1.240)
                    # (.075, 1.232) (.046, 1.238) (.024, 1.250) (.003, 1.254) -- 4-8 mm under the fold, a corner at each point
                    (("x", .1165, 1.252), ("x", .108, 1.2445), ("x", .100, 1.2422), ("x", .090, 1.2394), ("x", .075, 1.2382),
                     ("x", .060, 1.2394), ("x", .046, 1.2424), ("x", .036, 1.2462), ("x", .028, 1.2512), ("x", .020, 1.2565),
                     ("x", .012, 1.2620), ("x", .003, 1.2660)),
                    # princess lines, under the bust down to the waist and out into the side seam above the hip (B1n: from the
                    # seam's new place; B1b-B1m from (.066, 1.234))
                    (("x", .066, 1.2386), ("x", .060, 1.196), ("x", .055, 1.158), ("x", .058, 1.122), ("x", .070, 1.100), ("x", .086, 1.091),
                     ("x", .100, 1.088)),
                    # the leotard line: outer hip down to the crotch
                    (("phi", 0, 1.030), ("x", .130, 1.000), ("x", .090, .960), ("x", .050, .925), ("x", .020, .903), ("x", .004, .895)),
                    # from the thigh panel's foot to the knee pad (each end on another edge: no seam stops on the surface)
                    (("phi", 6, .662), ("x", .139, .641)),   # B1h: its end now under the wider pad
                    # B1h: the frame below the pad (as drawn): from the outer side, where the arc behind the knee passes, down to a
                    # point on the shin and up again to the pad's lower inner edge
                    # B1l (review_stage10 item 10): its end ON the pad's rim (B1i-B1k at (.104, .572): 1.4 mm inside the rim, where the
                    # pad's lower part sinks into the shin, the cord's end cap came up through the pad -- the dark dot)
                    (("phi", 0, .620), ("phi", 12, .590), ("x", .182, .556), ("x", .165, .524), ("x", .152, .505), ("x", .136, .514),
                     ("x", .119, .534), ("x", .104, .5685)),
                    # down the outer shin into the boot: B1h from the frame's point (B1g from under the pad: ("x", .137, .592),
                    # ("x", .150, .566), ("x", .172, .520), then as now from .460), ending on the boot's top seam
                    (("x", .152, .505), ("x", .174, .482), ("x", .190, .460), ("x", .200, .400), ("x", .200, .320), ("x", .196, .231)),
                    # behind the knee: an arc from under the pad's outer edge round the back to under its inner edge
                    (("x", .141, .602), ("phi", 0, .620), ("phi", -45, .636), ("bx", .124, .642), ("phi", -135, .636), ("phi", -180, .620),
                     ("x", .101, .602)),
                    # --- the back (without_hair_back.png): a V from the shoulder blades to the waist's centre
                    # B1l (review_stage10 item 5): from the blade insert's upper tip, as drawn (B1k from under the collar at (.043, 1.441), beside
                    # the raglan seam's start: two lines side by side on the yoke, and a hook where this one bent at (.050, 1.425))
                    (("bx", .060, 1.414), ("bx", .053, 1.396), ("bx", .047, 1.370), ("bx", .036, 1.300), ("bx", .022, 1.230), ("bx", .008, 1.180),
                     ("bx", .001, 1.166)),
                    # behind: the raglan seam from the collar outside the blade insert to the back of the armpit, and
                    # under the arm the short seam joining it to THE line (the shoulder's own mesh ends on these)
                    # B1l: ONE line (BACK_LINE): the raglan seam down the blade insert's edge, into the armpit and on down past the
                    # rib and waist panels round the hip. B1k: RAGLAN_BACK; (("bx", .1125, RAGLAN_LOW), ("phi", 0, RAGLAN_LOW),
                    # ("x", .1275, RAGLAN_LOW)) under the arm; and (("bx", .1122, 1.326), ("bx", .104, 1.280), ...) from beside the
                    # insert's tip -- the knot's three cords
                    BACK_LINE,
                    # round the seat (stage 4, as the back view draws it): from the hip down the seat's outer side, then
                    # level along its foot to the centre -- a broad U, not the stage-3 diagonal "heart"
                    (("phi", 0, 1.030), ("bx", .131, .990), ("bx", .128, .955), ("bx", .114, .928), ("bx", .090, .913), ("bx", .060, .906),
                     ("bx", .030, .902), ("bx", .004, .898)),
                    # centre back, below the zip
                    (("phi", -90, 0.898), ("phi", -90, 1.165)),
                    # down the back of the calf from the knee to the boot
                    (("bx", .124, .642), ("bx", .128, .600), ("bx", .138, .540), ("bx", .146, .460), ("bx", .156, .380), ("bx", .170, .300),
                     ("bx", .180, .231)),
                ),
                body_loops=(),   # stage 3: the knee's loops are gone (pads in front, an arc seam behind)
                # the sleeve: B1k the elbow seam, a closed ring ON the join of the sleeve and the shoulder's own surface
                # (B1j: a chevron at mid upper arm, ARM_SEAM)
                arm=("join",),
                # B1k: drawn on the shoulder's own surface: the seam over the foot of the deltoid, from the raglan seam in front
                # round the outside of the arm to the raglan seam behind
                shoulder=(DELTOID_SEAM,),
            ),
            mesh_panels=(
                # front of the shoulder: a thin curved slit from the collar's side down to the armpit's front
                dict(name="Shoulder", pts=(("x", .054, 1.424), ("x", .080, 1.402), ("x", .104, 1.370), ("x", .121, 1.336), ("x", .125, 1.324),
                                           ("x", .114, 1.334), ("x", .094, 1.362), ("x", .070, 1.392), ("x", .050, 1.414)),
                     shared=(3, 4), res=0.0015),   # B1k: its outer edge from (.121, 1.336) to the tip IS the raglan seam (one piped line);
                     # triangulated at 1.5 mm (4 mm triangles on the armpit's curves sank under the shoulder's own surface)
                # stage 3: the inserts as clean teardrops (bodysuit.Suit.teardrop: round end a -> point b, half-width w,
                # bow), placed from the three drawings. The ribs: from under the arm, down and forward to the waist,
                # the point on the side seam ("THE line")
                dict(name="Waist", drop=dict(a=("phi", -12, 1.264), b=("x", .080, 1.172), w=0.0125, bow=-0.08, cap=0.22)),
                # the waist's side: from the side of the waist down to the front of the hip, outside the side seam
                dict(name="Hip", drop=dict(a=("phi", -4, 1.144), b=("x", .118, 1.056), w=0.0105, bow=-0.10, cap=0.22)),
                # the back of the shoulder: a strip from the yoke down along the shoulder blade to the back of the armpit
                dict(name="Blade", pts=(("bx", .060, 1.414), ("bx", .084, 1.405), ("bx", .100, 1.378), ("bx", .110, 1.348), ("bx", .112, 1.332),
                                        ("bx", .107, 1.344), ("bx", .088, 1.374), ("bx", .068, 1.400)),
                     shared=(2, 4), res=0.0015),   # B1l: from its outer middle point (B1k (3, 4)); B1k: likewise behind (the raglan seam runs down its outer edge to its lower tip)
                # the back's sides at the ribs (from the side down and in towards the spine) and at the waist (a lens)
                dict(name="Back_Rib", drop=dict(a=("phi", -16, 1.268), b=("bx", .058, 1.204), w=0.0100, bow=0.10)),
                dict(name="Back_Waist", drop=dict(a=("bx", .076, 1.168), b=("bx", .102, 1.098), w=0.0075, bow=0.10, blunt=0.0)),
                # the outer thigh: a long almond from the hip down to the knee on the side of the thigh, dark grey: the
                # net over the suit, not skin. Stage 4: both edges bow well in from the outline at mid-thigh (front
                # view: the front edge ~3 cm inside, as drawn; back view: the back edge likewise), so neither reads
                # as a second outline beside the silhouette
                dict(name="Thigh", mat="mesh_thigh",
                     pts=(("phi", 2, 0.992), ("phi", 22, .955), ("phi", 42, .905), ("phi", 55, .845), ("phi", 58, .785), ("phi", 50, .730),
                          ("phi", 32, .690), ("phi", 6, .662), ("phi", -12, .690), ("phi", -30, .750), ("phi", -42, .820),
                          ("phi", -44, .870), ("phi", -36, .920), ("phi", -20, .962), ("phi", -6, .985))),
            ),
        ),
        boot=dict(
            ankle=(0.1885, 0.028), toe_out=12.0,   # stage 3: the drawings turn the feet out a little more
            # the foot, toe -> heel: (y along the foot, centre height, half-width, up, down); on a LOW heel
            # (Erik: about 3 cm), an almond toe
            foot=((-0.170, .015, .005, .007, .008), (-0.162, .017, .013, .011, .011), (-0.148, .019, .020, .016, .013),
                  (-0.128, .021, .026, .021, .015), (-0.102, .024, .031, .027, .017), (-0.076, .029, .033, .034, .019),
                  (-0.050, .036, .032, .041, .021), (-0.025, .044, .031, .045, .022), (0.000, .052, .031, .046, .023),
                  (0.022, .057, .032, .026, .024), (0.040, .059, .030, .018, .025), (0.052, .060, .020, .010, .018)),
            toe_cap_y=-0.118, panel_phi=24.0, sole=0.007, sole_end_y=-0.040,
            heel=(0.032, 0.046, 0.046),                    # (y, width, depth): a block heel
            shaft_z=(0.050, 0.240), shaft_ease=0.0012, shaft_thick=0.0020,
            feather=0.012,                                 # the top 12 mm thin into the leg: no step
            v_depth=0.030, v_half_angle=38.0,
            # stage 3: the boot as ONE implicit surface (bodysuit.build_boots_implicit), in the foot's frame
            # (y forward-negative from the ankle, z up): the foot's sections heel -> toe (y, centre z,
            # half-width, half-height); the underside line (heel seat, ball, toe); the top edge (z, front
            # dip, its half-angle rad); the shaft's ease over the leg (top, ankle); sole (thickness, its
            # back end y); the block heel (centre y, half-width, half-depth, floor/top scale) -- a low heel
            implicit=dict(
                foot=((0.034, 0.058, 0.0245, 0.0285), (0.004, 0.054, 0.0270, 0.0290), (-0.034, 0.048, 0.0285, 0.0250),
                      (-0.068, 0.037, 0.0300, 0.0200), (-0.098, 0.025, 0.0310, 0.0150), (-0.128, 0.019, 0.0265, 0.0115),
                      (-0.150, 0.016, 0.0175, 0.0090), (-0.160, 0.0150, 0.0085, 0.0060)),
                sole_line=((0.002, 0.032), (-0.088, 0.008), (-0.166, 0.013)),
                top=(0.232, 0.016, 0.95), ease=(0.0022, 0.0048), n=2.8, shaft_low=0.080, k=0.020,
                sole=(0.0045, 0.012), heel=(0.032, 0.0185, 0.0205, 0.82),
                suit_end=0.100, res=(0.0009, 0.0020), foot_k=0.012, smooth=12),
            # B1h (review_stage7 B1): the boot FITTED to the leg (bodysuit.FittedBoot), replacing `implicit` (kept above as
            # B1g's). In the foot's frame (y forward-negative from the ankle, z up). The shaft is the leg plus the leather,
            # its top (the drawn height) a seam ridge where it dives under the suit; a heel cup; the foot ONE sweep (rows:
            # y, half-width on the widest line, top line z, exponent) with a curved instep and an almond toe with a toe box;
            # a slim welt following the upper; the low block heel (the owner's 3 cm) under the heel seat
            fitted=dict(
                top=0.232, top_seam=0.2295, leather=0.0013, sink=(0.004, 0.0006), shaft_low=(0.090, 0.030),
                heel_cup=((0.0, 0.021, 0.056), (0.0300, 0.037, 0.046), 2.3), k=(0.035, 0.025),   # k: shaft-heel cup (0.022 left a dent behind the ankle bone), upper-foot
                foot=((0.000, .0280, 0.150, 2.2), (-0.025, .0292, 0.150, 2.2), (-0.035, .0297, 0.135, 2.2),
                      (-0.042, .0300, 0.106, 2.2), (-0.050, .0304, 0.086, 2.2), (-0.060, .0309, 0.073, 2.2), (-0.075, .0314, 0.062, 2.2),
                      (-0.090, .0316, 0.054, 2.2), (-0.105, .0310, 0.048, 2.15), (-0.120, .0285, 0.044, 2.1), (-0.140, .0235, 0.040, 2.0),
                      (-0.155, .0185, 0.036, 2.0), (-0.170, .0120, 0.030, 2.0)),
                tip=(-0.171, 0.022), widest=0.004,
                # the sole's underside: the heel seat (on the heel), the shank, the ball on the floor, the toe sprung
                sole=dict(line=((0.065, 0.027), (0.000, 0.027), (-0.010, 0.0262), (-0.030, 0.018), (-0.050, 0.007), (-0.070, 0.0012),
                                (-0.085, 0.0), (-0.110, 0.0), (-0.140, 0.0025), (-0.175, 0.0085)),
                          thick=0.005, welt=0.0012, into=0.0008),
                heel=dict(breast=-0.004, inset=0.0006, taper=0.08),
                toe_cap=(-0.122, 14.0), piping=(0.0009, 0.0002),
                # B1i (review_stage8 2): the ankle bones under the leather (x on the shaft's surface, y, z, height, radii across,
                # along, up): the outer lower and further back, the inner higher and further forward; a shallow hollow across the
                # back above the heel cup (1 mm: the leather is 1.3 mm over the suit there), so the cup rises out of it
                ankle=((0.0245, 0.003, 0.180, 0.0050, 0.009, 0.013, 0.016), (-0.0320, -0.006, 0.192, 0.0050, 0.009, 0.013, 0.016),
                       (-0.002, 0.037, 0.115, -0.0010, 0.014, 0.008, 0.018)),
                # the suit leg ends inside the boot at 0.10 as in B1g (0.6 mm inside the boot there; a higher end moved 20 suit
                # vertices at the shoulder by 0.009 mm: the body build is order-sensitive)
                suit_end=0.100, res=(0.0009, 0.0020), smooth=6),
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
HEAD_PLACEHOLDER = HEAD

# B2a, the face test: her real head, a stylised anime-style 3D face (`charkit/animehead.py`), at the A-pose drawings' size
# (chin 1.476 to crown 1.680 = 204 mm, cranium 150 mm wide; PROVENANCE B2a step 1). Points are (y, z) for the profile's
# front and back, (x, z) for the front view's side, in metres; they are hand-picked from the traced outlines: the cranium and
# the back of the head from the bald A-pose drawings (`without_hair.png`, `without_hair_side.png`), the face below the brow
# from the close-ups (`concept-face-front-v1.png` widths and eyes, `concept-face-side-v1.png` depths), both scaled so the
# eye line (1.5626) to the chin (1.476) is the A-pose drawing's.
FACE = dict(
    egg=dict(
        centre=(0.0, -0.005, 1.575),
        front=((-0.005, 1.680), (-0.027, 1.6759), (-0.0528, 1.6665), (-0.0728, 1.6524), (-0.0857, 1.6383), (-0.0940, 1.6242),
               (-0.0990, 1.6101), (-0.1010, 1.5960), (-0.1005, 1.5840), (-0.0975, 1.5720), (-0.0935, 1.5600), (-0.0910, 1.5480),
               (-0.0905, 1.5380), (-0.0915, 1.5280), (-0.0925, 1.5190), (-0.0905, 1.5100), (-0.0880, 1.5020), (-0.0840, 1.4960),
               (-0.0810, 1.4900), (-0.0795, 1.4840), (-0.0770, 1.4790), (-0.0700, 1.4762), (-0.0600, 1.4770), (-0.0480, 1.4810),
               (-0.0380, 1.4870), (-0.0280, 1.4960), (-0.005, 1.505)),
        back=((-0.005, 1.680), (0.0211, 1.6759), (0.0435, 1.6665), (0.0575, 1.6571), (0.0646, 1.6477), (0.0728, 1.6383),
              (0.0763, 1.6289), (0.0810, 1.6195), (0.0830, 1.6050), (0.0830, 1.5920), (0.0815, 1.5780), (0.0775, 1.5660),
              (0.0716, 1.5540), (0.0640, 1.5440), (0.0560, 1.5350), (0.0480, 1.5250), (0.0380, 1.5150), (0.0250, 1.5070),
              (-0.005, 1.505)),
        side=((0.0, 1.680), (0.0272, 1.6748), (0.0446, 1.6655), (0.0550, 1.6563), (0.0619, 1.6470), (0.0666, 1.6378),
              (0.0700, 1.6285), (0.0723, 1.6192), (0.0740, 1.6100), (0.0750, 1.6007), (0.0745, 1.5900), (0.0730, 1.5800),
              (0.0705, 1.5700), (0.0670, 1.5600), (0.0630, 1.5500), (0.0590, 1.5400), (0.0545, 1.5300), (0.0500, 1.5200),
              (0.0450, 1.5120), (0.0380, 1.5040), (0.0280, 1.4970), (0.0, 1.505)),
        # superellipse exponents round the head by elevation (deg): flat across the eyes, a V at the chin
        # B2b: rounder across the eyes and cheeks (B2a: 2.6, a flat mask on which the eyes read printed at 30-60 deg)
        n_front=((90, 2.0), (60, 2.2), (30, 2.3), (5, 2.35), (-10, 2.35), (-25, 2.2), (-40, 2.0), (-55, 1.8), (-70, 2.0), (-90, 2.0)),
        n_back=((90, 2.0), (40, 2.2), (0, 2.3), (-40, 2.2), (-90, 2.0)),
        # B2b: the profile's dents kept to the midline (`animehead.Egg`): smoothed along the elevation, blended round the head
        front_smooth=(12.0, 14.0),
    ),
    # the neck: the placeholder's rings inside the collar (it is fitted round them), slimmer above it as drawn
    neck=((1.404, .012, .040, .044, .044, 2.0), (1.443, .010, .040, .042, .044, 2.0), (1.474, .010, .040, .040, .046, 2.0),
          (1.482, .009, .0420, .0425, .0485, 2.05), (1.494, .008, .0378, .0365, .0460, 2.0), (1.510, .007, .0360, .0350, .0450, 2.0),
          (1.540, .004, .0360, .0350, .0420, 2.0), (1.570, .000, .0340, .0340, .0400, 2.0)),
    neck_k=0.010,
    # Gaussian displacements (centre (x >= 0, mirrored), sigmas, height m): the socket, the brow, the cheeks, the chin
    bumps=(((0.036, -0.088, 1.5625), (0.016, 0.012, 0.009), -0.0020),
           ((0.030, -0.096, 1.5775), (0.018, 0.010, 0.0045), 0.0012),
           ((0.044, -0.080, 1.5370), (0.014, 0.012, 0.011), 0.0020),
           ((0.000, -0.080, 1.4870), (0.009, 0.008, 0.006), 0.0012),
           ((0.000, -0.086, 1.4985), (0.010, 0.006, 0.0025), -0.0008),
           # B2b: the nose's root set deeper (the profile: B2a's nasion stood 4-6 mm in front of the drawn one at 1.549-1.564,
           # so the bridge read short and scooped; drawn, it runs straight from a deep root to the tip)
           ((0.000, -0.092, 1.5585), (0.0075, 0.010, 0.0065), -0.0065)),
    # the nose: small, slightly upturned; mostly its tip (front view: a highlight)
    nose=dict(bridge=((0.0, -0.0830, 1.5540), (0.0, -0.0960, 1.5395), 0.0018, 0.0030, 0.8),   # B2b: its root 4 mm back
              tip=((0.0, -0.0995, 1.5360), (0.0036, 0.0045, 0.0040)),
              column=((0.0, -0.0990, 1.5335), (0.0, -0.0925, 1.5220), 0.0026, 0.0032, 1.0),
              alae=((0.0030, -0.0880, 1.5262), (0.0012, 0.0012, 0.0012)),   # B2a: next to nothing (a dark diamond under the tip)
              k_tip=0.003, k_col=0.0025, k_alae=0.003, k=0.0060),
    # shading: the nose's bridge and sides (z from, to; across sigma; its tip (x, z) and radii kept; weight) and the lips (z
    # from, to; half-width; weight) shaded with the normals of the face without them (`Head.shading_normals`)
    # (B2a: the lips' weight 0 -- shaded flat they darkened the muzzle in profile)
    shading=dict(nose=((1.519, 1.556), 0.009, (0.0, 1.5362), (0.0035, 0.0035), 0.65), lips=((1.499, 1.519), 0.014, 0.0)),
    mouth=dict(upper=((0.0, -0.0898, 1.5130), (0.0100, 0.0032, 0.0030)), lower=((0.0, -0.0852, 1.5040), (0.0082, 0.0030, 0.0028)),
               k_between=0.001, k=0.0040, slot=((0.0, -0.0915, 1.5075), (0.0128, 0.0055, 0.00045)), slot_k=0.0006),
    # painted on the skin (front-projected): blush (centre (x, z), sigmas), mouth line (z, half-width, height), lower lash
    # line (x from, x to, below the lid, width), lid crease (x from, x to, above the lid, width), brow (x from, x to, z, arch,
    # width), nostrils ((x, z), sigmas), the nose tip's highlight (z, sigmas)
    paint=dict(blush=((0.044, 1.5445), (0.014, 0.0075)), mouth_line=(1.5075, 0.0125, 0.00045),
               low_lash=(0.024, 0.0515, 0.0006, 0.00035), crease=(0.021, 0.050, 0.0030, 0.00035),
               brow=(0.012, 0.052, 1.5795, 0.0025, 0.0014), nostril=(0.0040, 1.5240, (0.0012, 0.0007)),
               nose_tip=(1.5370, (0.0030, 0.0030))),
    # the ear (her left; mirrored) as a field (`animehead.Ear`): in the ear's own axes (up, back, out), metres; 47 mm high,
    # 27 wide, its top level with the brow's foot and its lobe with the nose's base, as both close-ups draw it
    ear=dict(centre=(0.066, 0.017, 1.553), tilt=12.0, flare=20.0, res=0.0004,
             plates=(((0.005, 0.0, 0.0), (0.019, 0.0135, 0.0060)), ((-0.014, -0.002, -0.0005), (0.009, 0.0085, 0.0050))), plate_k=0.006,
             cuts=(((0.003, 0.001, 0.0062), (0.0150, 0.0100, 0.0030), 0.0015), ((-0.004, -0.004, 0.0050), (0.0065, 0.0055, 0.0030), 0.0015)),
             adds=(((0.005, 0.003, 0.0040), (0.010, 0.0025, 0.002), 0.002), ((-0.004, -0.0125, 0.0035), (0.004, 0.0025, 0.0025), 0.002))),
)

# The eyes (her left; mirrored), in the front view: the lid curves (x, z) traced on the close-up, the iris, the eye
# surface's dome, the upper lash band.
EYES = dict(
    # B2b: re-traced on the front close-up's drawn opening (the sclera's edge, her left eye; PROVENANCE B2b step 2): 35 mm wide
    # (B2a: 42), nearly level (inner corner 2.6 mm under the eye line, the outer end round and blunt at 52 mm out, B2a's
    # pointed, raised corner at 59 mm made the cat-eye), the upper lid a heavier, rounder arch (+7.3 mm over 30-42 mm out)
    upper=((0.0180, 1.5600), (0.0200, 1.5630), (0.0220, 1.5658), (0.0240, 1.5679), (0.0270, 1.5693), (0.0310, 1.5699),
           (0.0360, 1.5700), (0.0410, 1.5698), (0.0450, 1.5692), (0.0480, 1.5684), (0.0500, 1.5675), (0.0512, 1.5664),
           (0.0520, 1.5649), (0.0524, 1.5628)),
    lower=((0.0180, 1.5600), (0.0200, 1.5586), (0.0220, 1.5573), (0.0250, 1.5556), (0.0280, 1.5541), (0.0310, 1.5534),
           (0.0350, 1.5530), (0.0390, 1.5531), (0.0420, 1.5536), (0.0450, 1.5547), (0.0480, 1.5560), (0.0500, 1.5574),
           (0.0514, 1.5591), (0.0522, 1.5609), (0.0524, 1.5628)),
    corner_k=0.0008, depth=0.005, depth_k=0.002, lid_k=0.0012,
    dome=dict(centre=(0.0352, 1.5615), axes=(0.0180, 0.0090), inset=0.0012, curve=0.0035, thick=0.0008, margin=0.0018),
    # B2b: the drawn iris is round, 19.4 x 20 mm, its centre 0.2 mm over the eye line (B2a: 1.5622, 20 x 22.4 mm: its
    # bottom sank 2 mm under the lower lid); the pupil 4.9 x 6.2 mm (`pupil`: half-axes in iris radii), straight ahead
    iris=dict(centre=(0.0337, 1.5628), radii=(0.0097, 0.0100), lid_shadow=0.006, pupil=(0.25, 0.31)),
    # B2b: the band along the new lid to 50.5 mm out, then a SHORT, nearly level flick to 56.8 mm (B2a: a long upswept wing
    # to 65 mm); heaviest over the outer third, as drawn; it covers the opening's top by 0.6 mm (the lid now traced on it)
    lash=dict(x=(0.0182, 0.0505), flick=(0.0568, 1.5682), width=((0.0, 0.0010), (0.3, 0.0030), (0.7, 0.0042), (0.88, 0.0040), (1.0, 0.0008)),
              thick=0.0010, below=0.0006, out=0.0004),
)

# The hair BLOCKOUT (`charkit/animehair.py`): a cap over the skull and swept locks. Angles about the egg's centre: beta the
# elevation, alpha the azimuth from the front (0) towards her left (90) and the back (180); every row mirrored. As drawn
# (`concept-face-*`, `with_hair*.png`, `concept.jfif`): a straight fringe to the brows, parted at the centre, its middle
# clumps dipping between the eyes; locks hanging in front of the ears (the ears showing behind them) down in front of the
# shoulders; the mass behind falling down the back to a V at about 1.20 m (the drawings: 1.17-1.23).
HAIR = dict(
    seed=5, body_z=0.95, body=("Suit_Body", "Suit_Shoulder", "Suit_Collar*", "Suit_Sleeve", "Suit_Mesh_Blade", "Suit_Mesh_Shoulder"),
    # the cap: hairline (alpha, z), thickness (front, crown, back), its fade, how deep it dives below the hairline
    cap=dict(line=((0, 1.640), (40, 1.630), (60, 1.605), (80, 1.585), (95, 1.590), (115, 1.594), (128, 1.560), (145, 1.530), (180, 1.515)),
             thick=(0.006, 0.012, 0.009), fade=0.006, under=0.009, res=0.0012),
    locks=(
        ("fringe", dict(n=6, b=(80.0,), a=(4.0, 50.0), ae=(3.0, 46.0), b_end=(-3.0, -0.5, 2.0, 3.0, 6.0, 10.0), off=0.0015, lift=0.004,
                        width=((0.0, 0.020), (0.6, 0.017), (0.85, 0.011), (1.0, 0.0005)), thick=((0.0, 0.005), (0.7, 0.0042), (1.0, 0.0008)))),
        ("crown", dict(n=5, b=(88.0,), a=(30.0, 150.0), ae=(80.0, 160.0), b_end=(40.0, 30.0, 20.0, 5.0, -10.0), off=0.0010,
                       width=((0.0, 0.030), (0.7, 0.028), (1.0, 0.004)), thick=((0.0, 0.004), (1.0, 0.002)))),
        ("side", dict(n=3, b=(55.0,), a=(50.0, 68.0), ae=(78.0, 88.0), b_end=(-14.0, -12.0, -10.0), off=0.0015, lift=0.003, hang=(0.25, 0.23, 0.20),
                      leave=(0.15, -0.35, -0.6), clear=0.005, len_jit=0.06,
                      width=((0.0, 0.026), (0.3, 0.025), (0.8, 0.015), (1.0, 0.0005)), thick=((0.0, 0.006), (0.5, 0.005), (1.0, 0.0010)))),
        # behind the ears, brought forward over the shoulders (the concept: locks in front of both shoulders)
        ("shoulder", dict(n=2, b=(60.0,), a=(96.0, 112.0), ae=(118.0, 126.0), b_end=(-14.0, -18.0), off=0.0015, hang=(0.24, 0.22),
                          leave=(0.55, -0.9, -0.3), clear=0.006, len_jit=0.05,
                          width=((0.0, 0.026), (0.4, 0.025), (0.85, 0.014), (1.0, 0.0006)), thick=((0.0, 0.006), (0.6, 0.005), (1.0, 0.0010)))),
        # the mass behind: the outer locks leave the skull at its widest and fall over the shoulder blades, the middle ones
        # from the nape to the V's point
        ("back_outer", dict(n=4, b=(66.0,), a=(112.0, 140.0), ae=(128.0, 146.0), b_end=(-14.0, -18.0, -22.0, -26.0), off=0.0015,
                            hang=(0.20, 0.23, 0.25, 0.27), leave=(0.45, 0.7, -0.3), clear=0.006, len_jit=0.04, jitter=1.0,
                            width=((0.0, 0.030), (0.5, 0.028), (0.85, 0.016), (1.0, 0.0008)), thick=((0.0, 0.007), (0.6, 0.006), (1.0, 0.0012)))),
        ("back", dict(n=5, b=(74.0,), a=(146.0, 177.0), ae=(150.0, 177.0), b_end=(-28.0, -32.0, -36.0, -38.0, -40.0), off=0.0015,
                      hang=(0.27, 0.29, 0.30, 0.31, 0.32), leave=(0.15, 1.2, -0.3), clear=0.006, len_jit=0.03, jitter=1.0,
                      width=((0.0, 0.030), (0.5, 0.028), (0.85, 0.016), (1.0, 0.0008)), thick=((0.0, 0.007), (0.6, 0.006), (1.0, 0.0012)))),
    ),
)

# The head table the build reads: the anime head (FACE, EYES, HAIR) at the concept's ratio, or a named variant (`--head NAME`).
# B2b (the owner, 2026-10-20): the x 1.20 head (the original concept's ratio to the body, PROVENANCE B2a step 1) is the
# default, scaled about the chin's height on the neck's axis (`animehead.Head`'s pivot, the neck left in the collar); `apose` is
# B2a's head at the A-pose drawings' size (chin to crown 204 mm); `placeholder` the bald one of B1a-B1n
HEAD_SCALE = 1.20
HEAD_APOSE = dict(kind="anime", face=FACE, eyes=EYES, hair=HAIR)
HEAD = dict(HEAD_APOSE, scale=HEAD_SCALE)
HEADS = dict(anime=HEAD, concept=HEAD, apose=HEAD_APOSE, placeholder=HEAD_PLACEHOLDER)


def head(name=None):
    return HEADS[name] if name else HEAD

# B2a: the face pictures (`scripts/face_views.py`, then `scripts/face_sheets.py`). `art`: orthographic shots in the close-ups'
# own frames (centre (x or y, z), the width in metres, the camera's side) so the render lies on the drawing pixel for pixel;
# `turn`: the head at azimuths (deg, 0 the front, + towards her left), portrait framing; `close`: aimed close-ups.
FACE_SHOTS = dict(
    art=dict(front=dict(centre=(-0.0054, 1.5564), width=0.2601, side="front", ref="concept-face-front-v1.png"),
             profile=dict(centre=(-0.0382, 1.5499), width=0.2652, side="right", ref="concept-face-side-v1.png")),
    turn=dict(azimuths=(0, 30, 60, 90, 135, 180), dist=1.15, lens=100.0, target=(0.0, -0.015, 1.560), elev=3.0, res=(720, 900)),
    close=dict(eyes=dict(cam=(0.0, -0.55, 1.565), target=(0.0, -0.08, 1.561), lens=135.0, res=(1200, 600)),
               eye_q=dict(cam=(0.30, -0.42, 1.570), target=(0.032, -0.075, 1.562), lens=135.0, res=(800, 600)),
               nose_mouth=dict(cam=(-0.08, -0.48, 1.53), target=(0.0, -0.09, 1.522), lens=135.0, res=(800, 600)),
               mouth_side=dict(cam=(-0.42, -0.30, 1.52), target=(0.0, -0.085, 1.515), lens=135.0, res=(800, 600))),
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
    "shoulder_back": (180.0, 30.0, 12.0, 1.1, 1.36, 85.0, (900, 900)),
    # B1i: aimed at her left wrist and cuff (B1h's `wrist` was a rig view aimed at the centre line: it showed the crotch)
    "wrist": dict(cam=(0.560, -0.360, 1.010), target=(0.303, 0.004, 0.915), lens=85.0, res=(900, 900)),
    "boot": (0.0, 30.0, 12.0, 1.0, 0.12, 85.0, (900, 900)),
    "flank": (0.0, 75.0, 5.0, 1.3, 1.18, 85.0, (900, 900)),
    "flank_back": (180.0, 60.0, 5.0, 1.3, 1.18, 85.0, (900, 900)),
    "knee": (0.0, 20.0, 5.0, 1.0, 0.55, 85.0, (900, 900)),
    "hand": (0.0, 90.0, 0.0, 0.75, 0.88, 85.0, (900, 900)),
    # aimed close-ups (charkit/suitbuild.render_aimed): the palm and thumb of her left hand from
    # behind and inside, between the hand and the thigh
    "boot_side": dict(cam=(0.95, -0.05, 0.16), target=(0.19, 0.0, 0.14), lens=85.0, res=(900, 900)),
    "armpit_back": dict(cam=(0.42, 0.62, 1.36), target=(0.13, 0.05, 1.28), lens=85.0, res=(900, 900)),
    "top_close": dict(cam=(0.0, -0.18, 2.45), target=(0.0, 0.0, 1.30), lens=50.0, res=(1000, 1000)),
    "hand_palm": dict(cam=(0.040, 0.380, 0.870), target=(0.305, 0.000, 0.860), lens=85.0, res=(900, 900)),
}

# The overlap bands for the sheet score: (name, z from, z to)
BANDS = (("above_waist", 1.140, 1.700), ("hip_knee", 0.600, 1.140), ("knees_down", 0.000, 0.600))


def palette(variant=None):
    p = dict(PALETTE)
    if variant:
        p.update(VARIANTS[variant].get("palette", {}))
    return p


def gloss(variant=None):
    """B1l: the gloss table for a variant (GLOSS with the variant's `gloss` entries over it)."""
    g = dict(GLOSS)
    if variant:
        g.update(VARIANTS[variant].get("gloss", {}))
    return g


# ------------------------------------------------------------------------ looks
# B2b (the owner, 2026-10-20): her high-resolution pictures in an anime / toon look, the whole character (`charkit/toonlook.py`,
# EEVEE); `pbr` is the physically-lit look of B1a-B2a (Cycles), kept to compare. Render-time only: the build never changes.
# Per material (its name less the prefix): `ramp` ((L, multiplier of the material's own colour), ...) over the light L (1 = a
# white surface square to the key), `interp` the ramp's blend; `spec` the streaks (a Glossy's light through `lo`..`hi`);
# `rim` (facing through `lo`..`hi`; `lit`: only on the lit side); `flat` how far the light is ignored (painted); `bump` scales
# the material's own relief. Colours by PALETTE name.
_SKIN_RAMP = ((0.00, (0.80, 0.62, 0.62)), (0.36, (0.82, 0.63, 0.62)), (0.44, (0.95, 0.70, 0.64)), (0.52, (1.0, 0.97, 0.96)),
              (1.20, (1.04, 1.03, 1.02)))
_LATEX = dict(ramp=((0.00, (1.0, 1.0, 1.0)), (0.40, (1.3, 1.3, 1.35)), (0.60, (3.4, 3.4, 3.7)), (1.30, (4.2, 4.2, 4.6))),
              # a broad soft sheen (the latex's grey lit side), then the crisp streaks
              spec=(dict(rough=0.32, lo=0.20, hi=0.75, colour="SUIT_RIM", amount=0.45),
                    dict(rough=0.08, lo=0.95, hi=1.10, colour="SHINE", amount=0.92)),
              rim=dict(lo=0.55, hi=0.85, blend=0.35, colour="SUIT_RIM", amount=0.55, lit=0.7))
LOOK_TOON = dict(
    engine=dict(samples=64, filter=1.2, shadow_rays=3, shadow_steps=12),
    world=dict(ambient=(0.92, 0.95, 1.0), strength=0.22),
    calib=1.0,
    # the light rig: a key from her upper front right (the art's), a rim from behind her left, two narrow strips and a small
    # top light that only the gloss sees (streaks); (name, kind, location, target, size or sun angle, energy, colour, options)
    lights=(("ToonKey", "SUN", (-0.85, -2.3, 3.3), (0.0, 0.0, 1.2), 2.5, 3.1416, (1.0, 0.98, 0.95), dict(filter=2.0)),
            ("ToonRim", "SUN", (1.6, 2.4, 1.9), (0.0, 0.0, 1.2), 4.0, 1.6, (0.92, 0.96, 1.0), dict(shadow=False)),
            ("ToonStripKey", "AREA", (-1.6, -2.2, 2.2), (0.0, 0.0, 1.2), (0.06, 0.9), 70.0, (1.0, 1.0, 1.0), dict(diffuse=0.0)),
            ("ToonStripSide", "AREA", (2.0, -1.2, 1.4), (0.0, 0.0, 1.1), (0.05, 1.0), 55.0, (1.0, 1.0, 1.0), dict(diffuse=0.0)),
            ("ToonStripTop", "AREA", (0.3, -1.5, 2.8), (0.0, 0.0, 1.3), (0.04, 0.30), 22.0, (1.0, 1.0, 1.0), dict(diffuse=0.0))),
    materials=dict(
        # the face lit nearly whole, as drawn: its shadow only where the simple head turns away from the key
        face=dict(ramp=((0.00, (0.82, 0.64, 0.63)), (0.20, (0.83, 0.65, 0.63)), (0.27, (0.95, 0.72, 0.66)), (0.34, (1.0, 0.97, 0.96)),
                        (1.20, (1.03, 1.02, 1.01)))),
        skin=dict(ramp=_SKIN_RAMP, bump=0.2, rim=dict(lo=0.6, hi=0.9, colour="SKIN_RIM", amount=0.35, lit=1.0)),
        nail=dict(ramp=_SKIN_RAMP, spec=dict(rough=0.2, lo=0.6, hi=0.8, colour="SHINE", amount=0.5)),
        eyes=dict(ramp=((0.0, (0.80, 0.80, 0.84)), (0.5, (1.0, 1.0, 1.0))), flat=0.85),
        lash=dict(ramp=((0.0, (1.0, 1.0, 1.0)), (1.0, (1.0, 1.0, 1.0))), flat=1.0),
        hair=dict(ramp=((0.00, (0.95, 0.95, 0.95)), (0.38, (1.0, 1.0, 1.0)), (0.55, (1.7, 1.6, 1.6)), (1.30, (1.9, 1.8, 1.8))),
                  bump=0.0, ring=dict(colour="HAIR_RING", lo=0.3, hi=0.6, amount=0.8)),
        suit=_LATEX, boot=_LATEX, pad=_LATEX,
        pad_rim=dict(_LATEX, ramp=((0.0, (0.8, 0.8, 0.8)), (0.6, (1.3, 1.3, 1.3)))),
        sole=dict(ramp=((0.0, (1.0, 1.0, 1.0)), (0.6, (2.0, 2.0, 2.0)))),
        metal=dict(ramp=((0.0, (0.45, 0.45, 0.48)), (0.5, (1.0, 1.0, 1.0)), (1.2, (1.3, 1.3, 1.3))),
                   spec=dict(rough=0.15, lo=0.6, hi=0.8, colour="SHINE", amount=0.8)),
        mesh=dict(ramp=((0.0, (0.75, 0.62, 0.62)), (0.45, (0.80, 0.66, 0.65)), (0.6, (1.15, 1.08, 1.05))),
                  spec=dict(rough=0.15, lo=0.85, hi=1.05, colour="SHINE", amount=0.5)),
        mesh_thigh=dict(ramp=((0.0, (0.8, 0.8, 0.8)), (0.6, (1.3, 1.3, 1.3))),
                        spec=dict(rough=0.15, lo=0.85, hi=1.05, colour="SHINE", amount=0.5)),
    ),
    # the face shaded as a simple head (`animehead.Head.toon_normals`): the egg's normals over the face's front, the nose
    # tip keeping some of its own
    face_normals=dict(weight=0.92, front=(-0.060, -0.015), z=(1.478, 1.492, 1.600, 1.625), tip=((0.0, 1.5362), (0.004, 0.004), 0.5),
                      lift=(0.35, 1.500, 1.535)),
    # the face casts as its simple shape (`animehead.Head.shadow_proxy`, shrunk 6 mm): the chin still shades the neck
    # (the hair's cap is also kept from casting: below the hairline it runs 9 mm inside the face, and its cut edge threw a
    # hard straight shadow across the cheek; the locks still cast the fringe's shadow on the forehead)
    shadow_proxy=dict(shrink=0.009, h=0.002, casters_off=("*_Head", "*_Hair_Cap")),
    # the outline over the nose and the lips left out when seen from the front (`animehead.Head.line_mask`, the amount;
    # `outline["front"]` how frontal): from the front the art draws no nose but its tip's shading, and the mouth as a line
    face_lines=dict(nose=(1.0, (1.530, 1.562)), lips=(0.7, (1.498, 1.520), 0.016)),
    # the hair shaded as one mass (`toonlook.radial_normals`): its normals towards the head's centre above the jaw
    hair_normals=dict(weight=0.75, z=(1.50, 1.58)),
    # the hair's highlight band (`toonlook.ring_attr`, degrees about the head's centre)
    ring=dict(objects=("*_Hair_Cap", "*_Hair_Locks"), beta=(40.0, 5.0, 4.0), jag=3.0, freq=11.0, back=170.0),
    floor=dict(lo=0.35, hi=0.55, amount=0.45, colour=(0.05, 0.05, 0.06)),
    # inked lines: an inverted hull `px` pixels wide at the subject; never on the eyes, lashes, piping, zip or nails
    outline=dict(px=1.6, exclude=("*_Eye", "*_Lash", "*_Piping*", "Suit_Zip_*", "Hand_Nails", "Suit_Knee_Rim"),
                 colours=dict(face="LINE_SKIN", skin="LINE_SKIN", hair="LINE_HAIR", suit="LINE_SUIT", boot="LINE_SUIT",
                              sole="LINE_SUIT", pad="LINE_SUIT", mesh="LINE_SUIT", mesh_thigh="LINE_SUIT", metal="LINE_SUIT")),
)
LOOKS = dict(toon=LOOK_TOON, pbr=None)
LOOK = "toon"


# ----------------------------------------------------------------------- posing
# B1m: the build posed for stills on a copy (`scripts/posed.py`, `charkit/suitpose.py`; the build never changes). Joints in
# metres (her LEFT side; the right is the mirror), measured on the B1l/B1m build: the hip on the thigh's axis (the sections'
# centres at 0.80 and 0.66 m, the inner thigh's fullness discounted), the knee and the ankle from LEGS, the arm from SLEEVE and
# the hand's own frame. Weight bands are (from, to) of a C2 ramp in metres: `trunk` the four heights where the pelvis hands over
# to the spine, the spine to the chest, the chest to the neck, the neck to the head; along a limb the ramp's span across the
# joint's plane; `arm_radius` how far from the arm's own chain a point still moves with the arm; `leg_mid` the fade to the
# mid-plane (the crotch moves with the pelvis). A pose gives bones their WORLD rotation, degrees about X (+: the top of an
# upright bone tips forward, a hanging one swings back), Y (+: her left side down; a hanging bone's end swings to her right),
# Z (+: the front turns to her left), applied in that order; an unlisted bone turns with its parent.
POSE = dict(
    joints=dict(pelvis=(0.0, -0.005, 0.98), spine=(0.0, -0.020, 1.08), chest=(0.0, -0.020, 1.22), neck=(0.0, 0.012, 1.44),
                head=(0.0, 0.000, 1.50),
                shoulder=(0.150, 0.032, 1.335), elbow=(0.198, 0.031, 1.140), wrist=(0.296, 0.007, 0.945),
                knuckle=(0.295, 0.008, 0.885), fingertip=(0.278, 0.010, 0.790),
                hip=(0.080, -0.005, 0.925), knee=(0.122, 0.010, 0.590), ankle=(0.188, 0.030, 0.115)),
    weights=dict(trunk=((1.00, 1.10), (1.16, 1.28), (1.41, 1.46), (1.48, 1.53)),
                 arm_root=(-0.02, 0.05), arm_radius=(0.05, 0.085), arm_radius_near=(0.16, 0.06, 0.16), arm_side=(0.04, 0.07), elbow=(-0.03, 0.03), wrist=(-0.015, 0.015),
                 leg_root=(-0.05, 0.05), leg_mid=(0.0, 0.05), knee=(-0.04, 0.04), ankle=(-0.025, 0.02)),
    # parts that turn whole, each side at the field's value at its centre
    rigid=("Suit_Knee_Pad", "Suit_Knee_Rim", "Suit_Zip_Pull*", "Suit_Zip_Stop*"),
    # B2a: parts moving wholly with one bone -- the hair rigid with the head this round
    attach={"Beatrice_Hair*": "head"},
    poses=dict(
        # concept.jfif: walking towards the camera, the weight on her RIGHT leg (its hip up and out, the pelvis shifted over it),
        # the left leg free (knee bent and forward, heel lifted), the shoulders tilted against the pelvis (the right one lower),
        # the right arm a little out and forward, the left hanging close beside the body, the head a little tilted
        concept=dict(
            offset=(-0.060, 0.0, 0.0), stand=("Boot_Sole", "R"), free=("Boot_Sole", "L"),
            rot={"pelvis": (0.0, 9.0, 5.0), "spine": (0.0, 8.0, 2.0), "chest": (0.0, -5.0, 0.0), "neck": (0.0, 0.0, 0.0),
                 "head": (0.0, 1.0, -2.0),
                 "upper_arm.L": (0.0, 3.0, 0.0), "forearm.L": (0.0, 20.0, 0.0),
                 "upper_arm.R": (-4.0, -2.0, 0.0), "forearm.R": (-12.0, -2.0, 0.0),
                 "thigh.R": (0.0, 1.0, 0.0), "shin.R": (0.0, -11.0, 0.0), "foot.R": (0.0, 0.0, -6.0),
                 "thigh.L": (-22.0, 8.0, -6.0), "shin.L": (18.0, 10.0, 0.0), "foot.L": (9.0, 0.0, 6.0)}),
    ),
    # shots: the camera in the world (x her left, -y her front); `rig` turns the studio rig (its lights) first
    shots=dict(
        concept=dict(cam=(-0.451, -2.560, 1.25), target=(-0.045, 0.0, 1.28), lens=90.0, res=(1024, 1536)),
        front=dict(cam=(0.0, -4.489, 1.174), target=(0.0, 0.0, 0.86), lens=85.0, res=(900, 1500)),
        front34=dict(cam=(-2.245, -3.888, 1.174), target=(0.0, 0.0, 0.86), lens=85.0, res=(900, 1500)),
        side=dict(cam=(-4.489, 0.0, 1.174), target=(0.0, 0.0, 0.86), lens=85.0, res=(900, 1500)),
        back34=dict(cam=(-2.245, 3.888, 1.174), target=(0.0, 0.0, 0.86), lens=85.0, res=(900, 1500), rig=180.0),
        # close-ups for checking the deformation, rendered on request only (`--views hips,seat`)
        hips=dict(cam=(-0.25, -1.55, 1.00), target=(-0.03, 0.0, 0.93), lens=85.0, res=(900, 900), dev=True),
        seat=dict(cam=(-0.55, 1.45, 1.00), target=(-0.03, 0.0, 0.92), lens=85.0, res=(900, 900), rig=180.0, dev=True),
        armpit=dict(cam=(0.75, -1.05, 1.25), target=(0.10, 0.0, 1.20), lens=85.0, res=(900, 900), dev=True),
        armpit_r=dict(cam=(-0.80, -0.95, 1.22), target=(-0.13, 0.0, 1.22), lens=85.0, res=(900, 900), dev=True),
        knee=dict(cam=(0.85, -1.15, 0.55), target=(0.05, -0.03, 0.45), lens=85.0, res=(900, 900), dev=True),
    ),
    # optional extra light setups (`--lights NAME`): the studio's broad lights scaled, plus narrow rectangles (name, location,
    # target, size, energy W, colour) that a glossy suit throws back as small hot streaks, as the concept's latex does
    lights=dict(
        hot=dict(studio_scale=0.45, world_scale=0.8, areas=(
            ("HotKey", (-1.6, -2.2, 2.2), (0.0, 0.0, 1.2), (0.06, 0.9), 45.0, (1.0, 0.98, 0.95)),
            ("HotSide", (2.0, -1.2, 1.4), (0.0, 0.0, 1.1), (0.05, 1.0), 40.0, (0.95, 0.97, 1.0)),
            ("HotTop", (0.3, -1.5, 2.8), (0.0, 0.0, 1.3), (0.12, 0.12), 25.0, (1.0, 1.0, 1.0)))),
        # B2a: a portrait light for the face pictures (`face_turn.jpg`): a big soft key above and to her right, a soft fill
        # from her left, a narrow rim behind her left shoulder
        portrait=dict(studio_scale=0.35, world_scale=0.8, areas=(
            ("PortraitKey", (-0.95, -1.35, 2.15), (0.0, -0.03, 1.57), (0.9, 0.9), 22.0, (1.0, 0.97, 0.94)),
            ("PortraitFill", (1.30, -1.10, 1.45), (0.0, -0.03, 1.57), (0.7, 0.7), 6.0, (0.95, 0.97, 1.0)),
            ("PortraitRim", (0.75, 1.05, 1.95), (0.0, 0.0, 1.57), (0.25, 0.9), 40.0, (0.92, 0.96, 1.0)))),
    ),
)
