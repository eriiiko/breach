"""Render char_04's head close-ups (front, side) with a known ortho frame so a
pixel maps to a height, and print the mesh bbox.
    blender -b -P inspect_head.py -- <char.glb> <out_prefix>
"""
import math, sys
import bpy
from mathutils import Vector
src, prefix = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
ob = next(o for o in bpy.context.scene.objects if o.type == "MESH")
vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
for i, a in enumerate("xyz"):
    print(f"BBOX {a}: {min(v[i] for v in vs):.3f} .. {max(v[i] for v in vs):.3f}")
print("PARENT", ob.parent, "VERTS", len(vs), "FACES", len(ob.data.polygons))
z1 = max(v.z for v in vs)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = scene.render.resolution_y = 800
w = bpy.data.worlds.new("w"); w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 1.0
scene.world = w
cd = bpy.data.cameras.new("c"); cd.type = "ORTHO"; cd.ortho_scale = 0.6
cam = bpy.data.objects.new("c", cd); scene.collection.objects.link(cam); scene.camera = cam
c = Vector((0, 0, z1 - 0.25))
print(f"FRAME centre z={c.z:.3f} scale 0.6 m over 800 px")
for name, d in {"front": (0, -3, 0), "side": (3, 0, 0), "back": (0, 3, 0)}.items():
    cam.location = c + Vector(d)
    cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True)
