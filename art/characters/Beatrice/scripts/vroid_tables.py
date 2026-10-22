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
    V_HAIR_SHADOW="#2e2324",   # the drawn hair's shadow (53-66, 43-51, 43-50), a touch deeper as a base under the toon ramp
    V_HAIR_MID="#5a4642",      # between the shadow and the mid tone (136, 107, 100)
    V_HAIR_LIT="#88706a",      # the mid tone, the base's top
    V_SKIN="#f2d0c2",          # the drawn skin (concept-face-front: the cheek away from the blush ~ (246, 214, 200))
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
        mesh=dict(beatrice.LOOK_TOON["materials"]["mesh"]),
        metal=dict(beatrice.LOOK_TOON["materials"]["metal"]),
    ),
    curvature=dict(materials=("suit",), smooth=6),
    outline=dict(px=1.6, exclude=("*_FaceLayers", "*_Zip*"),
                 colours=dict(face="LINE_SKIN", skin="LINE_SKIN", hair="LINE_HAIR", hair_back="LINE_HAIR", cloth="LINE_SUIT",
                              suit="LINE_SUIT", mesh="LINE_SUIT", metal="LINE_SUIT")),
)

# candidates: a head shot front and three-quarter (azimuth deg, + towards her left), framed on the eyes' bones
CANDIDATE_SHOTS = dict(azimuths=(0, 30), dist=1.25, lens=100.0, up=0.035, back=0.0, elev=3.0, res=(600, 750))
