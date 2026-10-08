"""Blender 4.5: build editable one-arm proof, export GLB, render evidence.

blender -b --factory-startup -P <this file> -- [--quick] [--animation]
Ordinary mesh + linear armature modifier. No Blender constraints are exported.
"""
from pathlib import Path
import argparse
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_math as arm

HOME = HERE.parent
OUT = HOME / "prototype"
PREVIEW = HOME / "previews" / "arm"
# canonical glTF Y-up -> Blender Z-up
B = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def material(name, color, roughness=.48):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def set_pose(rig, curl, contact, keyframe=None):
    points, rotations = arm.pose(curl, contact)
    for i, pb in enumerate(rig.pose.bones):
        rotation = Matrix((B @ rotations[i] @ B.T).tolist())
        rotation = rotation @ pb.bone.matrix_local.to_3x3()
        mat = rotation.to_4x4()
        mat.translation = Vector(B @ points[i])
        pb.matrix = mat
        # Children need the just-assigned parent's evaluated transform.
        bpy.context.view_layer.update()
        if keyframe is not None:
            pb.keyframe_insert("location", frame=keyframe)
            pb.keyframe_insert("rotation_quaternion", frame=keyframe)
            pb.keyframe_insert("scale", frame=keyframe)
    bpy.context.view_layer.update()


def build(root):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    data = arm.geometry()
    mesh = bpy.data.meshes.new("Arm_skin_and_reattachable_cups")
    mesh.from_pydata((data["vertices"] @ B.T).tolist(), [], data["faces"])
    mesh.update()
    obj = bpy.data.objects.new("Arm_mesh", mesh)
    bpy.context.collection.objects.link(obj)
    uv = mesh.uv_layers.new(name="Skin_atlas")
    for polygon, coords in zip(mesh.polygons, data["uvfaces"]):
        polygon.use_smooth = True
        for idx, coord in zip(polygon.loop_indices, coords):
            uv.data[idx].uv = coord
    # Recalculate outside normals, preserving the hollow sucker bowls.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    skin = material("Octopus_skin_atlas", arm.PALETTE["dorsal"], .46)
    tex = bpy.data.images.new("Arm_albedo_gloss", width=512, height=512, alpha=True)
    tex.pixels.foreach_set(arm.atlas().ravel())
    tex.filepath_raw = str(OUT / "arm_albedo_gloss.png")
    tex.file_format = "PNG"
    tex.save()
    # Blender treats alpha as coverage; Breach treats it as gloss. Keep the
    # identical RGB in an opaque studio image and splice gloss into the GLB.
    studio_pixels = arm.atlas()
    studio_pixels[..., 3] = 1
    tex = bpy.data.images.new("Arm_studio_albedo_opaque", width=512, height=512, alpha=False)
    tex.pixels.foreach_set(studio_pixels.ravel())
    tex.pack()
    node = skin.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = tex
    skin.node_tree.links.new(node.outputs["Color"], skin.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
    obj.data.materials.append(skin)

    rigdata = bpy.data.armatures.new("Arm_32_segments_plus_tip")
    rig = bpy.data.objects.new("Arm_rig", rigdata)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    obj.select_set(False)
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    parent = None
    for i in range(arm.SEGMENTS + 1):
        bone = rigdata.edit_bones.new(f"arm_{i:02d}")
        bone.head = Vector(B @ (arm.BASE + [i * arm.STEP, 0, 0]))
        bone.tail = Vector(B @ (arm.BASE + [(i + 1) * arm.STEP, 0, 0]))
        bone.parent = parent
        bone.use_connect = parent is not None
        parent = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.show_in_front = True
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for i in range(arm.SEGMENTS + 1):
        vg = obj.vertex_groups.new(name=f"arm_{i:02d}")
        for v, (joints, weights) in enumerate(zip(data["joints"], data["weights"])):
            for joint, weight in zip(joints, weights):
                if joint == i and weight > 0:
                    vg.add([v], float(weight), "REPLACE")
    modifier = obj.modifiers.new("Portable_linear_skin", "ARMATURE")
    modifier.object = rig
    modifier.use_deform_preserve_volume = False
    obj.parent = rig
    rig["prototype_note"] = "33 joints, 32 segments. Live controller: scripts/arm_math.py; glTF is not the controller."
    rig["length_metres"] = arm.LENGTH
    rig["oral_side"] = "Displayed upward for inspection; orientation is presentation only."

    # Two identical rest keys satisfy UnitModelRenderer's existing clip loader.
    # Runtime deliberately ignores this clip and supplies newly computed poses.
    set_pose(rig, 0, 0, 1)
    set_pose(rig, 0, 0, 2)
    rig.animation_data.action.name = "Idle_Loop"
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, 2
    bpy.context.scene.frame_set(1)
    obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(OUT / "octopus_arm.glb"), export_format="GLB",
                             use_selection=True, export_yup=True, export_animations=True,
                             export_animation_mode="ACTIVE_ACTIONS", export_force_sampling=True,
                             export_skins=True, export_def_bones=True, export_materials="EXPORT")
    # The canonical helper preserves alpha=gloss despite glTF's opaque export.
    sys.path.insert(0, str(root / "art" / "characters"))
    from charkit.glbtex import replace_image
    replace_image(OUT / "octopus_arm.glb", "baseColor", (OUT / "arm_albedo_gloss.png").read_bytes())
    rig.animation_data_clear()
    # Full editable loop in .blend, all generated through the shared controller.
    for frame in range(1, 218, 3):
        c, r = arm.cycle_weights((frame - 1) / 24)
        set_pose(rig, c, r, frame)
    rig.animation_data.action.name = "Shared_curve_inspection_loop"
    for curve in rig.animation_data.action.fcurves:
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"
    for source in [HERE / "arm_math.py", Path(__file__)]:
        bpy.data.texts.load(str(source))
    np.savez_compressed(OUT / "arm_geometry.npz", vertices=data["vertices"],
                        joints=data["joints"], weights=data["weights"], groups=data["groups"])
    triangles = sum(len(f) - 2 for f in data["faces"])
    stats = dict(length_m=arm.LENGTH, joints=arm.SEGMENTS + 1,
                 deform_segments=arm.SEGMENTS, vertices=len(data["vertices"]),
                 triangles=triangles, cups=len(data["cups"]),
                 eight_arms_triangles=8 * triangles, eight_arms_joints=8 * (arm.SEGMENTS + 1))
    (OUT / "geometry_budget.json").write_text(json.dumps(stats, indent=2) + "\n")
    return obj, rig


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def studio():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 760
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (.16, .16, .16)
    scene.view_settings.view_transform = "AgX"
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -.012))
    floor = bpy.context.object
    floor.name = "Studio_floor_not_exported"
    floor.data.materials.append(material("Studio_floor", arm.PALETTE["floor"]))
    for name, loc, power, size, color in [
        ("Key", (.3, -1.2, 2.4), 120, 2.0, (1., .87, .75)),
        ("Fill", (-.5, 1.3, 1.8), 80, 1.7, (.65, .81, 1.)),
        ("Edge", (1.7, .7, 1.5), 100, 1.0, (1., .64, .4)),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.shape, data.size, data.color = power, "DISK", size, color
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = loc
        aim(obj, (.5, -.25, .15))
    data = bpy.data.cameras.new("Arm_camera")
    camera = bpy.data.objects.new("Arm_camera", data)
    scene.collection.objects.link(camera)
    camera.location = (1.55, -2.2, 2.3)
    data.type, data.ortho_scale = "ORTHO", 1.68
    aim(camera, (.53, -.21, .16))
    scene.camera = camera
    # A small freestanding flat contact plane, x=1.0 m. Never exported.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(arm.CONTACT_X + .015, -.41, .25))
    target = bpy.context.object
    target.name = "Geometric_contact_plane_x_1m_not_exported"
    target.dimensions = (.03, .5, .50)
    target.data.materials.append(material("Contact_plane", arm.PALETTE["target"], .55))
    bevel = target.modifiers.new("Rounded_back_edges", "BEVEL")
    bevel.width, bevel.segments = .006, 3
    target.hide_render = True
    scene.render.fps = 24
    scene.frame_start, scene.frame_end = 1, 216
    return camera, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Read-only checkout containing charkit.glbtex")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--animation", action="store_true")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    obj, rig = build(args.root.resolve())
    camera, target = studio()
    scene = bpy.context.scene
    scene.frame_set(73)
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "octopus_arm.blend"))
    rig.animation_data_clear()
    parity = {}
    for name, curl, contact in [("extended", 0, 0), ("curl", 1, 0), ("contact", 0, 1), ("dorsal", 0, 0)]:
        set_pose(rig, curl, contact)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        evaluated_mesh = evaluated.to_mesh()
        actual = np.array([v.co[:] for v in evaluated_mesh.vertices]) @ B
        data = arm.geometry()
        expected = arm.skin(data["vertices"], data["joints"], data["weights"], *arm.pose(curl, contact))
        parity[name] = float(np.max(np.linalg.norm(actual - expected, axis=1)))
        evaluated.to_mesh_clear()
        centre = (actual.min(axis=0) + actual.max(axis=0)) / 2
        target_point = B @ centre
        camera.location = Vector(target_point) + Vector((-.8 if name == "contact" else 1.1, -2.4, 2.6))
        aim(camera, target_point)
        camera.data.ortho_scale = 1.52 if name != "curl" else .86
        rig.rotation_euler.x = math.pi if name == "dorsal" else 0
        rig.location.z = .30 if name == "dorsal" else 0
        target.hide_render = name != "contact"
        scene.render.filepath = str(PREVIEW / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    (OUT / "blender_skin_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if args.animation:
        rig.rotation_euler.x, rig.location.z = 0, 0
        camera.location = (1.55, -2.2, 2.3)
        camera.data.ortho_scale = 1.68
        aim(camera, (.53, -.21, .16))
        target.hide_render = True
        scene.render.resolution_x, scene.render.resolution_y = 640, 440
        scene.cycles.samples = 12
        frames = PREVIEW / "frames"
        frames.mkdir(exist_ok=True)
        for i in range(54):
            c, r = arm.cycle_weights(i * arm.PERIOD / 54)
            set_pose(rig, c, r)
            scene.render.filepath = str(frames / f"{i:03d}.png")
            bpy.ops.render.render(write_still=True)
    print("ARM_BUILD_OK", OUT, flush=True)


if __name__ == "__main__":
    main()
