"""Read-only import assessment; never overwrites the canonical glTF."""
import bpy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'source'
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(ROOT / 'assets/models/marine/AnimationLibrary_Godot_Standard.gltf'))
report = {'source': 'assets/models/marine/AnimationLibrary_Godot_Standard.gltf', 'meshes': [], 'armatures': []}
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        report['meshes'].append({'name': obj.name, 'vertices': len(obj.data.vertices), 'polygons': len(obj.data.polygons), 'dimensions': list(obj.dimensions)})
    if obj.type == 'ARMATURE':
        report['armatures'].append({'name': obj.name, 'matrix_world': [list(r) for r in obj.matrix_world], 'bones': [{'name': b.name, 'parent': b.parent.name if b.parent else None, 'head': list(b.head_local), 'tail': list(b.tail_local), 'matrix_local': [list(r) for r in b.matrix_local]} for b in obj.data.bones]})
(OUT / 'canonical_rig_inspection.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps({'meshes': report['meshes'], 'rigs': [{'name': r['name'], 'bones': len(r['bones'])} for r in report['armatures']]}))
from mathutils import Vector
for obj in bpy.context.scene.objects:
    if obj.name == 'Icosphere': obj.hide_render = True
    if obj.type == 'ARMATURE': obj.data.pose_position = 'REST'
scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.color_type = 'SINGLE'
scene.display.shading.single_color = (0.55, 0.6, 0.65)
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
bpy.ops.object.camera_add(location=(0,-5,0.9))
cam = bpy.context.object
cam.rotation_euler = (Vector((0,0,0.9))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO'
cam.data.ortho_scale=2.35
scene.camera=cam
scene.render.resolution_x=800
scene.render.resolution_y=800
scene.render.resolution_percentage=100
preview=OUT.parent / 'previews'
preview.mkdir(exist_ok=True)
scene.render.filepath=str(preview/'canonical_base_assessment.png')
bpy.ops.render.render(write_still=True)
