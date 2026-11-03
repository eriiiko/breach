"""Swap char_04's albedo for a recoloured one, make the visor glossy, export a
.glb and render front / 3-4 / back.

    blender -b -P apply_render.py -- <in.glb> <albedo.png> <visor_mask.npy> <out.glb> <render_prefix>
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]
src, albedo, vmask, out_glb, prefix = a
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
mat = bpy.data.materials["Material_0"]
nodes = mat.node_tree.nodes
base = next(n for n in nodes if n.type == "TEX_IMAGE" and n.image.name == "Image_0")
base.image = bpy.data.images.load(albedo)
base.image.name = "albedo_recolor"
base.image.pack()

# Visor gloss: Image_1 is glTF metallic-roughness (G = roughness, B = metal).
mr = bpy.data.images["Image_1"]
w, h = mr.size
px = np.array(mr.pixels[:], dtype=np.float32).reshape(h, w, 4)
mask = np.load(vmask)[::-1]                 # PNG row 0 = top, Blender row 0 = bottom
px[mask, 1] = 0.08
px[mask, 2] = 0.0
mr.pixels[:] = px.ravel()
mr.pack()

bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB")

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x, scene.render.resolution_y = 700, 900
scene.view_settings.view_transform = "AgX"
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.45, 0.47, 0.5, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
scene.world = world
for rot, e in (((55, 0, 25), 3.0), ((60, 0, 200), 1.2)):
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = e
    ob = bpy.data.objects.new("sun", ld)
    ob.rotation_euler = [math.radians(v) for v in rot]
    scene.collection.objects.link(ob)
cd = bpy.data.cameras.new("cam")
cd.type = "ORTHO"
cd.ortho_scale = 2.3
cam = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
ob = next(o for o in scene.objects if o.type == "MESH")
zs = [(ob.matrix_world @ v.co).z for v in ob.data.vertices]
z0, z1 = min(zs), max(zs)
mid = Vector((0, 0, (z0 + z1) / 2))
for name, pos in {"front": (0, -5, 0.05), "front34": (3, -4, 0.6), "back": (0, 5, 0.05)}.items():
    cam.location = mid + Vector(pos)
    cam.rotation_euler = (mid - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True)
# helmet close-up
cd.ortho_scale = 0.7
head = Vector((0, 0, z1 - 0.2))
cam.location = head + Vector((0.9, -3, 0.1))
cam.rotation_euler = (head - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.render.filepath = f"{prefix}_helmet.png"
bpy.ops.render.render(write_still=True)
