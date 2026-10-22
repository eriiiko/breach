"""Beatrice on a VRoid base (the VRoid spike, 2026-10-22): the tables. The mechanism is `charkit/vrmbase.py`, the look
`charkit/toonlook.py` (with `beatrice.LOOK_TOON` as the model), the driver `scripts/vroid.py`.

The references (the owner, 2026-10-22: ONLY these four count for this build): `concept.jfif` (the original),
`concept-face-front-v1.png`, `concept-face-side-v1.png`, `concept-upper-body-side-v2.png`. The A-pose sheets
(`with_hair*.png`, `without_hair*.png`) are NOT used for it.

  BASES      the CC0 VRM files used (committed under `art/characters/Beatrice/vroid/`) and the candidates compared
  RECOLOUR   per material kind: the VRoid painting recoloured to the drawn palette (`vrmbase._recolour`)
  BODY       proportions (bone scales) and the suit's cut (signed fields on the rest body, `suit_fields`)
  LOOK_VROID the toon look's table for the VRoid materials (`toonlook.apply`)
  POSE_VROID the concept's stance on the VRM skeleton, and the shots
"""
import copy
import os

import numpy as np

import beatrice

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
VROID_DIR = os.path.join(ROOT, "vroid")
PREFIX = "vroid"

# ------------------------------------------------------------------------------------------------ the bases
# The candidates (step 1): the OpenGameArt "VRoid Studio CC0 models" files, read from the scratch download folder (they are
# not all committed: see PROVENANCE, "VRoid spike", for the licence each one carries). `cc0`: CC0 by BOTH VRoid's own help
# page and the file's embedded VRM licence.
# (name, path under DOWNLOADS, licence status): "cc0" = CC0 by both; "page" = CC0 by VRoid's page (beta AvatarSample_1 =
# Sendagaya Shibu) but the file says "Other" (the VRoid Hub terms); "other" = the file says "Other" and VRoid's page does not
# list it as CC0. Only "cc0" files are committed. Sakurada Fumiriya (on the OGA page) is a MALE model: not a candidate.
CANDIDATES = (
    ("base_female", "base_female/x/Base_Female.vrm", "cc0"),
    ("hairsample_female", "hairsample_female/x/HairSample_Female.vrm", "cc0"),
    ("avatarsample_d_darkness", "avatarsample_d_darkness/x/AvatarSample_D_Darkness.vrm", "cc0"),
    ("avatarsample_d", "avatarsample_d_0/x/AvatarSample_D.vrm", "page"),
    ("avatarsample_e", "avatarsample_e/x/AvatarSample_E.vrm", "other"),
    ("avatarsample_f", "avatarsample_f/x/AvatarSample_F.vrm", "other"),
    ("avatarsample_g", "avatarsample_g/x/AvatarSample_G.vrm", "other"),
    ("sendagaya_shino", "sendagaya_shino/x/Sendagaya Shino.vrm", "other"),
)
LICENCE_NOTE = dict(cc0="CC0 (VRoid's page AND the file's meta)",
                    page="CC0 by VRoid's page (beta AvatarSample_1), the file's meta says 'Other' -- not committed",
                    other="the file's meta says 'Other' (VRoid Hub terms), not on VRoid's CC0 list -- not committed")
DOWNLOADS = "C:/tmp/beatrice_vroid/dl"

# ------------------------------------------------------------------------------------------------ palette
# The drawn colours (B2e sampled the hair; PROVENANCE B2e step 1) and Beatrice's palette for the rest
PALETTE = dict(beatrice.PALETTE)
PALETTE.update(
    V_HAIR_SHADOW="#221a1b",   # the drawn hair's shadow (53-66, 43-51, 43-50), a touch deeper as a base under the toon ramp
    V_HAIR_MID="#45363a",      # between the shadow and the mid tone (136, 107, 100)
    V_HAIR_LIT="#76625e",      # the mid tone, the base's top
    V_SKIN="#f2d0c2",          # the drawn skin (concept-face-front: the cheek away from the blush ~ (246, 214, 200))
    V_ZIP="#46464c",           # the zip: a dark metal under the latex look's lift (METAL read white)
    V_IRIS_DARK="#2c3346", V_IRIS="#5f6f8c", V_IRIS_LIGHT="#a4b2cc",   # blue-grey as drawn (the front close-up's iris)
)

# ------------------------------------------------------------------------------------------------ the toon look
_FLAT = dict(ramp=((0.0, (1.0, 1.0, 1.0)), (1.0, (1.0, 1.0, 1.0))), flat=1.0, alpha=True)
_EYE = dict(ramp=((0.0, (0.86, 0.86, 0.90)), (0.5, (1.0, 1.0, 1.0))), flat=0.85, alpha=True)
_HAIR = dict(ramp=((0.00, (0.80, 0.78, 0.80)), (0.30, (0.84, 0.82, 0.84)), (0.42, (1.20, 1.17, 1.15)), (1.20, (1.38, 1.33, 1.30))),
             bump=0.0, rim=dict(lo=0.55, hi=0.85, blend=0.3, colour="HAIR_EDGE", amount=0.25, lit=1.0), alpha=True)
LOOK_VROID = dict(
    {k: v for k, v in beatrice.LOOK_TOON.items() if k in ("engine", "world", "calib", "lights", "floor")},
    materials=dict(
        face=dict(beatrice.LOOK_TOON["materials"]["face"]),
        skin=dict(beatrice.LOOK_TOON["materials"]["skin"], bump=1.0),
        eye_white=_EYE, iris=_EYE, eye_hi=_FLAT, eye_extra=_FLAT, eyeline=_FLAT, lash=_FLAT, brow=_FLAT, mouth=_FLAT,
        hair=_HAIR, hair_back=_HAIR,
        cloth=dict(ramp=((0.0, (0.70, 0.70, 0.74)), (0.40, (0.74, 0.74, 0.78)), (0.5, (1.0, 1.0, 1.0)), (1.2, (1.1, 1.1, 1.1))),
                   alpha=True),
        suit=dict(beatrice.LOOK_TOON["materials"]["suit"], alpha=True),
        mesh=dict(beatrice.LOOK_TOON["materials"]["mesh"], alpha=True),
        metal=dict(beatrice.LOOK_TOON["materials"]["metal"]),
    ),
    curvature=dict(materials=("suit",), smooth=6),
    outline=dict(px=1.6, exclude=("*_FaceLayers", "*_Zip*"), masks=dict(suit="suit", mesh="insert"),
                 colours=dict(face="LINE_SKIN", skin="LINE_SKIN", hair="LINE_HAIR", hair_back="LINE_HAIR", cloth="LINE_SUIT",
                              suit="LINE_SUIT", mesh="LINE_SUIT", metal="LINE_SUIT")),
)

# candidates: a head shot front and three-quarter (azimuth deg, + towards her left), framed on the eyes' bones
CANDIDATE_SHOTS = dict(azimuths=(0, 30), dist=1.25, lens=100.0, up=0.035, back=0.0, elev=3.0, res=(600, 750))

# ------------------------------------------------------------------------------------------------ B: the whole VRoid figure
BASE_B = os.path.join(VROID_DIR, "HairSample_Female.vrm")   # face, body and hair: one CC0 file (PROVENANCE, "VRoid spike")

# The sample's painting recoloured to the drawn palette (per material kind; `vrmbase._recolour`)
RECOLOUR = dict(
    face=dict(tint="V_SKIN"),        # the face's painting (blush, lips, nose) kept, its mean moved to the drawn skin
    skin=dict(flat="V_SKIN"),        # the body: one colour (its texture paints underwear, which the leg openings would show)
    hair=dict(ramp=((0.0, "V_HAIR_SHADOW"), (0.55, "V_HAIR_MID"), (1.0, "V_HAIR_LIT"))),
    iris=dict(ramp=((0.0, "V_IRIS_DARK"), (0.45, "V_IRIS"), (1.0, "V_IRIS_LIGHT")), keep=0.0),
)

# The hair: the sample's cat ears and twin tails removed, its back and face-framing locks lengthened (`vrmbase.drape`)
HAIR_B = dict(
    ears=dict(zmax=1.655, materials=("hair_02", "HAIR_02")),   # pieces reaching above the crown, and the ears' inner card
    tails=dict(zmin=1.385, xmin=0.055),                         # pieces hanging below the nape, off the midline: the tails
    back=dict(z0=1.50, end=(1.12, 0.035), flare=0.10, max_r=0.15, axis=(0.0, 0.02), clear=0.012, push_below=1.46),
    side=dict(z0=1.47, end=(1.21, 0.020), flare=0.04, max_r=0.16, axis=(0.0, 0.0), clear=0.010, push_below=1.44),
    # re-weighted to the trunk by height (the B2e lesson: rigid with the head to its tips, a nod sends long hair into the body)
    trunk=(("C_Head", None), ("C_Neck", (1.43, 0.03)), ("C_UpperChest", (1.34, 0.05)), ("C_Chest", (1.20, 0.06))),
)


def _resolve(spec):
    """Palette names in a recolour spec -> hex."""
    out = {}
    for k, v in spec.items():
        if k in ("tint", "flat"):
            out[k] = PALETTE.get(v, v)
        elif k == "ramp":
            out[k] = tuple((p, PALETTE.get(c, c)) for p, c in v)
        else:
            out[k] = v
    return out


def recolour():
    return {k: _resolve(v) for k, v in RECOLOUR.items()}

# shots of the T-posed build (development): (cam, target, lens, res)
SHOTS = dict(
    t_front=dict(cam=(0.0, -6.0, 0.85), target=(0.0, 0.0, 0.85), lens=85.0, res=(900, 1100)),
    t_side=dict(cam=(-6.0, 0.0, 0.85), target=(0.0, 0.0, 0.85), lens=85.0, res=(900, 1100)),
    t_back=dict(cam=(0.0, 6.0, 0.85), target=(0.0, 0.0, 0.85), lens=85.0, res=(900, 1100)),
    t_head=dict(cam=(-0.35, -1.25, 1.48), target=(0.0, 0.0, 1.40), lens=85.0, res=(800, 1000)),
    t_head_back=dict(cam=(0.45, 1.25, 1.48), target=(0.0, 0.0, 1.35), lens=85.0, res=(800, 1000)),
)


# ------------------------------------------------------------------------------------------------ B: the suit, as cut layers
# The four references: a glossy black latex LEOTARD with long sleeves to the wrist and a stand collar with a zip to the lower
# belly, cut high over the hips (bare hip and outer thigh, concept.jfif and concept-upper-body-side-v2.png), glossy black
# thigh-high stockings below (the same latex), fine mesh inserts showing the skin: one from the collarbone down to the front
# of the armpit, one at the side of the ribs under the bust running down towards the waist (both drawings), thin piping on
# the cut edges. As layers over the VRoid body (`vrmbase.shell`, cut by these signed fields, metres, + = covered), all in the
# rest T-pose, from the skeleton's landmarks. The boots and the knee pads are NOT drawn by these references (both crop above
# the knee): the stockings run to the toes; no heels (left out).
SUIT_B = dict(
    offset=0.0016,             # the latex layer off the skin
    sleeve_end=0.020,          # the sleeve ends this far short of the hand bone (the wrist)
    collar=(-0.012, 0.014),    # the collar's top: head bone z + this at the back, this much lower at the front
    gusset=0.028,              # the crotch's half-width
    hip_half=0.125,            # where the cut reaches its top, from the midline
    cut_top=(0.050, 0.000),    # the cut's top above the hips bone: front, back (the back covers more of the seat)
    cut_pow=(0.80, 1.70),      # the cut's curve: front, back
    stocking=(0.095, 0.020),   # the stockings' top below the crotch, rising this much towards the outside
    zip=(0.0021, 0.055),       # the zip's half-width; its lower end above the hips bone
    inserts=(  # (from (x, z), to (x, z), half-width, facing: (min -n_y, min n_x)) on her left; mirrored
        ((0.062, 1.318), (0.118, 1.240), 0.0075, (0.05, -1.0)),
        ((0.100, 1.160), (0.062, 1.060), 0.0100, (-0.20, -0.30)),
    ),
    piping=0.0016, piping_colour="#3b3a40", zip_colour="V_ZIP",
    # the stand collar (`vrmbase.collar`): standing off the neck to just under the jaw, as all four references draw it
    collar_geo=dict(z=(1.338, 1.418), gap=0.006, flare=0.010, front_drop=0.012, thick=0.0018),
    net=("#2a2224", 0.0042, 0.30),   # the inserts' net: thread colour, cell (m), thread share (the drawing: fine, dark)
)


def _seg_dist2(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def suit_fields(vrm, body_rest, spec=SUIT_B):
    """The signed fields (metres, + = the layer covers) on rest vertices V (world) with normals N: `suit` (the latex),
    `insert` (the mesh panels), `zip`."""
    hips, head, hand = vrm.bone("C_Hips"), vrm.bone("C_Head"), vrm.bone("L_Hand")
    B = body_rest
    crotch = B[(np.abs(B[:, 0]) < 0.006) & (B[:, 2] > 0.5) & (B[:, 2] < hips[2])][:, 2].min()
    S = spec

    def f(V, N):
        x, y, z = np.abs(V[:, 0]), V[:, 1], V[:, 2]
        front = np.clip(-y / 0.05 + 0.5, 0.0, 1.0)
        sleeve = (hand[0] - S["sleeve_end"]) - x
        collar = head[2] + S["collar"][0] - S["collar"][1] * front - z
        s = np.clip((x - S["gusset"]) / (S["hip_half"] - S["gusset"]), 0.0, 1.0)
        zf = crotch + (hips[2] + S["cut_top"][0] - crotch) * s ** S["cut_pow"][0]
        zb = crotch + (hips[2] + S["cut_top"][1] - crotch) * s ** S["cut_pow"][1]
        leg = z - (zf * front + zb * (1.0 - front))
        stock = (crotch - S["stocking"][0] + S["stocking"][1] * np.clip(x / 0.13, 0.0, 1.0)) - z
        suit = np.maximum(np.minimum(np.minimum(sleeve, collar), leg), stock)
        P2 = np.stack([x, z], 1)
        ins = np.full(len(V), -1.0)
        for a, b, r, (fy, fx) in S["inserts"]:
            nx = N[:, 0] * np.sign(V[:, 0] + 1e-9)
            ok = (-N[:, 1] >= fy) & (nx >= fx)
            ins = np.maximum(ins, np.where(ok, r - _seg_dist2(P2, a, b), -1.0))
        zp = np.minimum(S["zip"][0] - x, np.minimum(collar, z - (hips[2] + S["zip"][1])))
        zp = np.where(y < 0.0, zp, -1.0)
        return dict(suit=suit, insert=ins, zip=zp)
    return f, crotch

SHOTS["t_face"] = dict(cam=(0.0, -0.80, 1.47), target=(0.0, 0.0, 1.47), lens=100.0, res=(600, 700))
# expression trials (the VRoid face's own shape keys, by name end)
FACE_TESTS_1 = dict(
    neutral={},
    a={"Fcl_MTH_Neutral": 1.0},
    b={"Fcl_MTH_Neutral": 1.0, "Fcl_EYE_Angry": 0.3, "Fcl_BRW_Angry": 0.35},
    c={"Fcl_MTH_Neutral": 1.0, "Fcl_EYE_Sorrow": 0.3, "Fcl_BRW_Sorrow": 0.25},
    d={"Fcl_MTH_Neutral": 1.0, "Fcl_EYE_Angry": 0.2, "Fcl_EYE_Sorrow": 0.2, "Fcl_BRW_Angry": 0.3, "Fcl_EYE_Highlight_Hide": 0.5},
    e={"Fcl_MTH_Angry": 0.3, "Fcl_EYE_Natural": 0.5, "Fcl_BRW_Angry": 0.25},
    f={"Fcl_MTH_Neutral": 1.0, "Fcl_EYE_Close": 0.18, "Fcl_BRW_Angry": 0.3},
    g={"Fcl_ALL_Neutral": 1.0},
)

# ------------------------------------------------------------------------------------------------ B: face, proportions, pose
# The expression (the VRoid face's shape keys, by name end; FACE_TESTS picked "d"/"f"): the mouth closed, the lids a little
# heavier and the brows lower and straighter (the drawings' calm, level look), the eye highlight smaller (drawn: small)
EXPRESSION = {"Fcl_MTH_Neutral": 0.7, "Fcl_MTH_Angry": 0.3, "Fcl_EYE_Angry": 0.25, "Fcl_EYE_Close": 0.16, "Fcl_BRW_Angry": 0.3,
              "Fcl_EYE_Highlight_Hide": 0.5}
# A longer lower face (the VRoid face is a child's: eye line to chin about 0.37 of the head's height; the drawings' about
# 0.41): every face vertex below the eye line moved down by `k` x its depth below it, fading out over `fade` below the chin
FACE_STRETCH = dict(k=0.10, fade=0.035)
# Bone scales (pose space): the bust fuller, as both drawings (the base's is small); children keep their own size
SCALES = {"J_Sec_L_Bust1": (1.28, 1.28, 1.28), "J_Sec_R_Bust1": (1.28, 1.28, 1.28)}
# the mouth's inside cards (teeth) left out: the expression keeps the mouth closed, and the card showed a tooth through it
DROP_FACE = ("mouth",)

# Poses: absolute world rotations from the T-pose rest (deg about world X, Y, Z; order "XYZ" unless given), as
# `beatrice.POSE`'s convention (X+: an upright bone's top tips forward, a hanging one swings back; Y+: her left side
# down / a hanging bone's end to her right; Z+: the front turns to her left). The arms are brought down about Y first.
_ARMS_DOWN = {"L_UpperArm": (0.0, 78.0, 0.0, "YXZ"), "L_LowerArm": (0.0, 84.0, 0.0, "YXZ"),
              "R_UpperArm": (0.0, -78.0, 0.0, "YXZ"), "R_LowerArm": (0.0, -84.0, 0.0, "YXZ")}
POSE_B = dict(
    stand=dict(rot=_ARMS_DOWN, stand="R"),
    # concept.jfif (beatrice.POSE["poses"]["concept"], carried over): weight on her RIGHT leg, the pelvis shifted and tilted
    # over it, the left knee bent forward, the shoulders tilted against the pelvis, the right arm a little out and forward,
    # the left close beside the body, the head tilted to her right shoulder, chin down
    concept=dict(stand="R", rot={
        "C_Hips": (0.0, 9.0, 5.0), "C_Spine": (0.0, 8.0, 2.0), "C_Chest": (0.0, -5.0, 0.0), "C_Neck": (0.0, 0.0, 0.0),
        "C_Head": (10.0, -9.5, -2.0),
        "L_UpperArm": (0.0, 80.0, 0.0, "YXZ"), "L_LowerArm": (0.0, 96.0, 0.0, "YXZ"),
        "R_UpperArm": (-5.0, -74.0, 0.0, "YXZ"), "R_LowerArm": (-14.0, -86.0, 0.0, "YXZ"),
        "R_UpperLeg": (0.0, -2.0, 0.0), "R_LowerLeg": (0.0, -11.0, 0.0), "R_Foot": (0.0, 0.0, -6.0),
        "L_UpperLeg": (-22.0, 2.0, -6.0), "L_LowerLeg": (18.0, 10.0, 0.0), "L_Foot": (9.0, 0.0, 6.0)}),
)
# fingers relaxed: each finger joint bent this much (deg)
FINGER_CURL = dict(Index=(8.0, 14.0, 10.0), Middle=(10.0, 16.0, 12.0), Ring=(12.0, 18.0, 12.0), Little=(14.0, 20.0, 14.0),
                   Thumb=(0.0, 8.0, 8.0))

# The concept framing: the camera direction of `beatrice.POSE["shots"]["concept"]` (from her right front, 90 mm), its
# distance and aim solved so the posed eyes' midpoint and the crotch land where concept.jfif has them (1024 x 1536)
# Framings on the drawings (`vroid.framed_camera`): the camera's direction (from the target), lens, resolution = the
# drawing's, and where two landmarks lie in the drawing (pixels, measured on the drawings' irises, chin and crotch)
FRAMES = dict(
    concept=dict(dir=(-0.451, -2.560, -0.03), lens=90.0, res=(1024, 1536), marks=("eyes", "crotch"),
                 px=((498.0, 328.0), (470.0, 1495.0))),
    # concept-upper-body-side-v2.png: a profile seen from her RIGHT (she faces the picture's right), head to mid-thigh
    side_v2=dict(dir=(-1.0, 0.0, 0.02), lens=90.0, res=(1024, 1536), marks=("eyes", "crotch"), px=((565.0, 245.0), (540.0, 1300.0)),
                 fit="vertical"),
    # the close-ups: the front (eyes' midpoint, chin) and the profile from her right
    art_front=dict(dir=(0.0, -1.0, 0.0), lens=200.0, res=(1254, 1254), marks=("eyes", "chin"), px=((648.0, 588.0), (650.0, 990.0))),
    # (in profile the eye bones sit deep in the skull: the height is fitted on eyes to chin, the aim across on the nose's tip)
    art_side=dict(dir=(-1.0, 0.0, 0.0), lens=200.0, res=(1254, 1254), marks=("eyes", "chin"), px=((765.0, 560.0), (785.0, 925.0)),
                  fit="vertical", aim_mark="nose", aim_px=(900.0, 690.0)),
)
FACE_VIEWS = (0, 30, 60, 90)
SHOTS.update(
    p_front=dict(cam=(0.0, -6.0, 0.85), target=(0.0, 0.0, 0.82), lens=85.0, res=(700, 1100)),
    p_34=dict(cam=(-3.0, -5.2, 0.95), target=(0.0, 0.0, 0.82), lens=85.0, res=(700, 1100)),
    p_side=dict(cam=(-6.0, 0.0, 0.85), target=(0.0, 0.0, 0.82), lens=85.0, res=(700, 1100)),
    p_back=dict(cam=(0.0, 6.0, 0.85), target=(0.0, 0.0, 0.82), lens=85.0, res=(700, 1100)),
)
# the face views (B2e's `face_turn` framing: 100 mm, portrait), on the posed eyes' midpoint
FACE_SHOT = dict(azimuths=FACE_VIEWS, dist=1.30, lens=100.0, up=0.0, back=-0.015, elev=3.0, res=(720, 900), transparent=True)

# round 2 (on the chosen eyes): the mouth -- the VRoid neutral mouth turns up at the corners (a cat smile); the drawings' is level
_E = {"Fcl_EYE_Angry": 0.25, "Fcl_EYE_Close": 0.16, "Fcl_BRW_Angry": 0.3, "Fcl_EYE_Highlight_Hide": 0.5}
FACE_TESTS = dict(m0=dict(_E, Fcl_MTH_Neutral=1.0), m1=dict(_E, Fcl_MTH_Neutral=0.6, Fcl_MTH_Sorrow=0.4),
                  m2=dict(_E, Fcl_MTH_Neutral=0.7, Fcl_MTH_Angry=0.3), m3=dict(_E, Fcl_MTH_Sorrow=0.6),
                  m4=dict(_E, Fcl_MTH_Neutral=0.6, Fcl_MTH_Down=0.4), m5=dict(_E, Fcl_MTH_Neutral=1.0, Fcl_BRW_Sorrow=0.3))
SHOTS["t_face"] = dict(cam=(-0.30, -0.74, 1.47), target=(0.0, 0.0, 1.455), lens=100.0, res=(600, 700))
