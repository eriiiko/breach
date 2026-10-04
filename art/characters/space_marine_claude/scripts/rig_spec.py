"""The space marine's rig spec for `charkit/rig.py`: its joints, its hard gear, its clip tweak.

Every joint is read off `marine.py`'s own geometry (metres, the character's frame: faces -Y,
+X is its LEFT, feet on z = 0), never eyeballed from a render. Left side only; the rig stage
mirrors x for the right. The character is modelled arms-down: these are the joints of THAT pose.
"""
import numpy as np

from marine import ARM, HAND_B, HAND_L, LEG, T_ELBOW, WRIST, HC, HCZ, bw, sole_top, Y_TOE


def _leg_centre(z):
    """Centre line of the trouser loft at height z."""
    return LEG.frames(LEG.t_at_z(z))[0][0]


def _thigh_line(z, z0=0.565, z1=0.80):
    """The thigh's centre line (through the loft centres at the knee and at mid-thigh), at height z."""
    a, b = _leg_centre(z0), _leg_centre(z1)
    return a + (b - a) * (z - z0) / (z1 - z0)


def _hand_point(l, w=0.0, b=0.0):
    """A point in the glove's frame (`parts.hand`): l down the fingers, w towards the thumb, b out of the back."""
    W = np.cross(HAND_L, HAND_B)
    return WRIST + l * HAND_L + w * W + b * HAND_B


HIP_Z = 0.95  # just above the crotch gusset's top (0.935); the belt starts at 1.097
KNEE_Z = 0.565  # the knee pad's centre (`build_armor`: LEG.t_at_z(0.565)); the knee flex band is 0.468-0.662
ANKLE_Z = 0.115  # the ankle disc on the boot shaft (`build_boots`: SHAFT at z 0.115)
BALL_Y = -0.154  # rear edge of the toe cap (centre -0.198, half-length 0.044), boot-local y

JOINTS = dict(
    # centre line
    pelvis=(0.0, 0.0, 0.94),  # the torso loft's axis (y = 0), between the hip joints' height and the crotch
    neck_base=(0.0, -0.004, 1.534),  # bottom of the collar yoke (YOKE first ring: z 1.534, y -0.004)
    head_pivot=(0.0, -0.012, 1.650),  # inside the neck ring (NECK_RING: 1.600-1.667, y -0.012), below the seal at 1.668
    head_top=(0.0, float(HC[1]), float(HC[2] + HCZ)),  # helmet dome top (HC + HCZ = 1.880)
    # left arm
    shoulder=(0.234, 0.010, 1.448),  # centre of the pauldron's ellipsoid (`build_armor`: sh = OnEllipsoid((0.234, 0.010, 1.448), ...))
    elbow=tuple(ARM.frames(T_ELBOW)[0][0]),  # sleeve loft centre at its elbow ring (ring 4)
    wrist=tuple(WRIST),  # marine.WRIST, the glove's origin
    knuckle=tuple(_hand_point(0.094, 0.011, -0.002)),  # base of the middle finger (`parts.hand`: DIGITS["Middle"])
    # left leg
    hip=tuple(np.append(_thigh_line(HIP_Z)[:2], HIP_Z)),
    knee=tuple(np.append(_leg_centre(KNEE_Z)[:2], KNEE_Z)),
    ankle=tuple(bw(0.0, -0.004, ANKLE_Z)),  # boot shaft's axis (SHAFT rings: bw(0, -0.004, z))
    ball=tuple(bw(0.0, BALL_Y, sole_top(BALL_Y) + 0.010)),  # on the foot loft's axis
    toe_tip=tuple(bw(0.0, Y_TOE, sole_top(Y_TOE) + 0.010)),
)

# The boots' toe-out (`marine.py`: BX, BY = the boot frame turned by D(13)). The feet are fitted
# flat with this yaw; the ball and toe-tip joints above only set the foot and toe lengths.
FOOT_YAW_DEG = 13.0

# Hard gear moves rigidly with one bone. Each skin vertex is labelled with the high-res source part
# nearest to it; the first rule whose prefix matches that part's name decides. `{s}` is the side
# (L where x > 0). None keeps the automatic (bone-heat) weights.
GEAR = (
    ("Helmet", "DEF-head"), ("Visor", "DEF-head"),
    ("Pack_Hose", None),
    ("Pack", "DEF-spine.003"),
    ("Chest", "DEF-spine.003"),
    ("Collar", "DEF-spine.003"), ("Neck_Ring", "DEF-spine.003"),
    ("Neck_Seal", None),
    ("Belt", "DEF-hips"), ("Pouch", "DEF-hips"),
    ("Cargo", "DEF-thigh.{s}"),
    ("Pauldron", "DEF-upper_arm.{s}"),
    ("Elbow", "DEF-forearm.{s}"),
    ("Knee", "DEF-shin.{s}"),
    ("Glove_Cuff", "DEF-forearm.{s}"),
    ("Glove", "DEF-hand.{s}"),
    ("Boot_Shaft", "DEF-shin.{s}"), ("Boot_Ankle", "DEF-shin.{s}"), ("Boot_Heel", "DEF-shin.{s}"),
    ("Boot", "DEF-foot.{s}"),
)

# Extra upper-arm abduction (degrees, away from the body) added to every clip: the clips were
# authored for a slim mannequin.
ABDUCTION_DEG = 0.0
