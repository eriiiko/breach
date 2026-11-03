"""Carve char_04's head out from inside its protective collar (the collar is
kept) and seat a beak helmet in the opening.

    blender -b -P fit_helmet.py -- <char.glb> <helmet.glb> <helmet_tex.png> <out_dir> <tag>
        [--carve cx,cy,cz,rx,ry,rz] [--seat y,z,width]

Writes <out_dir>/char_04_headless.glb (once per carve), helmet_beak_<tag>.glb
(seated in the character's frame, its own asset), char_04_beak_<tag>.glb (both
together, for viewing) and renders r_<tag>_{front,front34,side,helmet}.png.
The character faces -y, z up, 1.9 m tall, centred on the origin.
"""
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
char_glb, helm_glb, helm_tex, out_dir, tag = a[:5]
out = Path(out_dir)
def opt(name, default):
    return [float(v) for v in a[a.index(name) + 1].split(",")] if name in a else default
cx, cy, cz, rx, ry, rz = opt("--carve", [0.0, -0.06, 0.795, 0.105, 0.11, 0.115])
sy, sz, width = opt("--seat", [-0.06, 0.80, 0.23])

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# --- the character: delete every vertex inside the head ellipsoid ---------
bpy.ops.import_scene.gltf(filepath=char_glb)
body = next(o for o in scene.objects if o.type == "MESH")
body.name = "char_04"
bm = bmesh.new()
bm.from_mesh(body.data)
mw = body.matrix_world
doomed = [v for v in bm.verts
          if (((mw @ v.co).x - cx) / rx) ** 2 + (((mw @ v.co).y - cy) / ry) ** 2
          + (((mw @ v.co).z - cz) / rz) ** 2 < 1.0]
print("CARVE verts", len(doomed), "of", len(bm.verts))
bmesh.ops.delete(bm, geom=doomed, context="VERTS")
bm.to_mesh(body.data)
bm.free()

# --- the helmet: beak to -y, scaled to width, seated ---------------------
bpy.ops.import_scene.gltf(filepath=helm_glb)
helm = next(o for o in scene.objects if o.type == "MESH" and o is not body)
helm.name = "helmet_beak"
bpy.ops.object.select_all(action="DESELECT")
helm.select_set(True)
bpy.context.view_layer.objects.active = helm
helm.rotation_mode = "XYZ"
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
vs = [v.co for v in helm.data.vertices]
low = [v for v in vs if v.z < -0.3]
front = 1 if sum(v.x for v in low) / len(low) > 0 else -1   # the beak hangs low and forward
print("HELMET beak towards", "+x" if front > 0 else "-x")
helm.data.transform(Matrix.Rotation(-math.pi / 2 * front, 4, "Z"))   # beak -> -y
vs = [v.co for v in helm.data.vertices]
lo = Vector([min(v[i] for v in vs) for i in range(3)])
hi = Vector([max(v[i] for v in vs) for i in range(3)])
s = width / (hi.x - lo.x)
centre = (lo + hi) / 2
helm.data.transform(Matrix.Translation(Vector((0, sy, sz))) @ Matrix.Scale(s, 4) @ Matrix.Translation(-centre))
print(f"HELMET scale {s:.4f}  size {(hi - lo).x*s:.3f} x {(hi - lo).y*s:.3f} x {(hi - lo).z*s:.3f} m")

mat = helm.active_material
bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
img_node = bsdf.inputs["Base Color"].links[0].from_node
img_node.image = bpy.data.images.load(helm_tex)
img_node.image.name = f"helmet_beak_{tag}"
img_node.image.pack()
mat.name = f"helmet_beak_{tag}"

def export(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True)
headless = out / "char_04_headless.glb"
if not headless.exists():
    export([body], headless)
export([helm], out / f"helmet_beak_{tag}.glb")
export([body, helm], out / f"char_04_beak_{tag}.glb")

# --- renders ---------------------------------------------------------------
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
cam = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
shots = {"front": (Vector((0, 0, 0)), 2.3, (0, -5, 0.05)),
         "front34": (Vector((0, 0, 0)), 2.3, (3, -4, 0.6)),
         "side": (Vector((0, -0.05, 0.72)), 0.7, (5, 0, 0.05)),
         "helmet": (Vector((0, -0.05, 0.75)), 0.7, (2.2, -3, 0.6))}
for name, (aim, scale, d) in shots.items():
    cd.ortho_scale = scale
    cam.location = aim + Vector(d)
    cam.rotation_euler = (aim - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out / f"r_{tag}_{name}.png")
    bpy.ops.render.render(write_still=True)
