"""Kessler's rig spec for `charkit/rig.py`: his joints, his hard gear, his clip tweak.

Every joint is measured on the scaled source mesh itself (character frame: faces -Y, +X is his
LEFT, feet on z = 0, `kessler_source.HEIGHT_M` tall), never eyeballed from a render: ray probes
through the mesh (a limb's centre = the midpoint of its two surfaces along each axis, at each
station), cross-section fits of the arm's two segments, the crotch's underside from rays cast up
the centre line, and the soles' own outline (2026-10-26; the numbers are in the comments). He is
modelled in a T-pose with the arms ~10 deg below horizontal; these are the joints of THAT pose,
which `rig.py` makes his rest. Left side only; the rig stage mirrors x for the right.
If HEIGHT_M changes, every number here scales with it (they were measured at 1.85 m).
"""

JOINTS = dict(
    # centre line
    pelvis=(0.0, -0.05, 0.84),  # 0.12 above the crotch's underside (rays up the centre line: z 0.70-0.72 at y -0.1..0.05)
    neck_base=(0.0, -0.06, 1.49),  # the collar's bottom: the arms' top surface meets the yoke at z ~1.49-1.56
    head_pivot=(0.0, -0.08, 1.575),  # under the helmet's bottom rim (z 1.55-1.57), below the helmet's centre (y -0.10)
    head_top=(0.0, -0.10, 1.85),  # helmet dome top (front y -0.28, back y ~0.08)
    # left arm (station centres: mid of the front/back and top/bottom surfaces at each x)
    shoulder=(0.24, 0.015, 1.43),  # the upper arm's root centre at x 0.30-0.36 is (y 0.01-0.03, z 1.43), inside the torso's side (x 0.25)
    elbow=(0.59, -0.005, 1.352),  # where the sleeve (centres level at z 1.36, y 0.00 over x 0.48-0.58) meets the gauntlet (x 0.60, turning forward)
    wrist=(0.84, -0.12, 1.318),  # the gauntlet's end / glove cuff (centre at x 0.84: y -0.119, z 1.321)
    knuckle=(0.925, -0.15, 1.285),  # base of the fingers (glove centre x 0.90-0.92: y -0.14, z 1.31; fingers curl down to z 1.21)
    # left leg
    hip=(0.15, -0.065, 0.82),  # on the thigh's centre line (x 0.16 at z 0.62-0.66), 0.10 above the crotch
    knee=(0.195, -0.085, 0.50),  # behind the knee pad's centre (pad z 0.42-0.58); leg centre x 0.19-0.20, calf back y 0.02-0.03
    ankle=(0.205, -0.03, 0.10),  # in the boot shaft, 9 cm in front of the heel's back (0.179, 0.058) along the foot line
    ball=(0.24, -0.185, 0.04),  # 25 cm along the foot line from the heel
    toe_tip=(0.27, -0.29, 0.035),  # the toe's front (sole outline: 0.269, -0.294)
)

# The boots' toe-out: the heel -> toe line of the soles (z < 6 cm) is 14.3 deg off -Y (PCA of the
# sole outline: 19-21 deg, pulled by the wide toe). The feet are fitted flat with this yaw.
FOOT_YAW_DEG = 15.0

# Hard gear moves rigidly with one bone (see the marine's spec for the rule format). The parts are
# `kessler_source.REGIONS`' names (`<Region>_L/_R`); "Suit" (everything else) keeps the bone heat.
GEAR = (
    ("Helmet", "DEF-head"), ("Visor", "DEF-head"), ("Lamp", "DEF-head"),  # the head turns as one block (Erik)
    ("Pack_Hose", None),  # the hoses from the pack to the arms stretch with the arms
    ("Pack", "DEF-spine.003"),
    ("Chest", "DEF-spine.003"),
    ("Coil", "DEF-hips"), ("Belt", "DEF-hips"), ("Plate", "DEF-hips"),
    ("Shin_Plate", "DEF-shin.{s}"),
    ("Gauntlet", "DEF-forearm.{s}"),
    ("Glove", "DEF-hand.{s}"),
    ("Boot_Shaft", "DEF-shin.{s}"),
    ("Boot_Toe", "DEF-toe.{s}"),  # the sole bends at the ball (and the floor clamp needs toe vertices)
    ("Boot", "DEF-foot.{s}"),
)

# Extra upper-arm abduction (deg) for every clip. The clips hold the upper arms ~15 deg from
# vertical in Idle (slim mannequin); Kessler's torso with its pouches is +-0.30 m wide at the belt,
# his sleeves 0.20 m thick from a shoulder at x 0.24, so his arms need ~35 deg to clear it.
ABDUCTION_DEG = 20.0

FLOOR_CLAMP = True  # the boots kept out of the floor (as for the workers)

TEST_POSES = (("Idle_Loop", 0), ("Walk_Loop", 0), ("Walk_Loop", 16), ("Pistol_Shoot", 5), ("Death01", -1),
              ("Crouch_Idle_Loop", 0))
