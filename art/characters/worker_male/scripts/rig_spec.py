"""This worker's rig spec for `charkit/rig.py`, DERIVED from its tables (`worker.py`) by the shared
`charkit/workergame.py::spec`: joints from DIMS / HEAD / HAIR, the boots' toe-out, the arm
abduction from the modelled upper-arm angle, the rigid-part table (face cut line, toe seam) and the floor clamp of the feet."""
import worker
import workergame

_S = workergame.spec(worker)
JOINTS = _S["JOINTS"]
FOOT_YAW_DEG = _S["FOOT_YAW_DEG"]
GEAR = _S["GEAR"]
ABDUCTION_DEG = _S["ABDUCTION_DEG"]
MODELLED_ARM_DEG = _S["MODELLED_ARM_DEG"]
TEST_POSES = _S["TEST_POSES"]
FLOOR_CLAMP = _S["FLOOR_CLAMP"]
