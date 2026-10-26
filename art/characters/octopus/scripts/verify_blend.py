"""Reopen the saved editable file and compare its actual timeline deformation.

blender -b prototype/octopus_arm.blend -P scripts/verify_blend.py
Protects saved rig weights, action, axis conversion and modifier configuration:
deleting a weight group, scale animation or corrupt pose keys must fail.
"""
from pathlib import Path
import json
import sys
import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_math as arm

B = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
d = arm.geometry()
obj = bpy.data.objects["Arm_mesh"]
rig = bpy.data.objects["Arm_rig"]
assert rig.animation_data.action is not None
assert obj.modifiers["Portable_linear_skin"].object == rig
assert not obj.modifiers["Portable_linear_skin"].use_deform_preserve_volume
key_errors, between_errors = [], []
for frame in range(1, 217):
    bpy.context.scene.frame_set(frame)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    actual = np.array([v.co[:] for v in mesh.vertices]) @ B
    expected = arm.skin(d["vertices"], d["joints"], d["weights"],
                        *arm.pose(*arm.cycle_weights((frame - 1) / 24)))
    error = float(np.linalg.norm(actual - expected, axis=1).max())
    (key_errors if (frame - 1) % 3 == 0 else between_errors).append(error)
    evaluated.to_mesh_clear()
    assert np.isfinite(actual).all()
assert max(key_errors) < 1e-5, max(key_errors)
# The saved editable action has three-frame samples. Between them Blender's
# local-bone interpolation approximates the live controller; bound that error.
assert max(between_errors) < .003, max(between_errors)
report = dict(file=str(Path(bpy.data.filepath)), timeline_frames=216,
              saved_key_pose_max_error_m=max(key_errors),
              between_keyframes_max_error_m=max(between_errors),
              note="Saved Blender keys every three frames; ordinary editable interpolation between keys. The runtime and rendered frames evaluate the continuous controller directly.")
(HERE.parent / "prototype" / "blender_saved_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("SAVED_BLEND_OK", json.dumps(report))
