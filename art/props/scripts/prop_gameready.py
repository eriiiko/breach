"""Game-ready step for a generated prop (Krea .glb -> game .glb).

    blender -b -P prop_gameready.py -- <in.glb> <out.glb> <preview_dir>
        (--height H | --length L) [--front -y] [--tris 4000] [--tex 1024]

  1. imports the generator's .glb (one mesh, one material) and turns it so its
     front faces -Y (``--front`` names the axis the front faces in the source);
  2. scales it to real size -- ``--height`` metres tall, or ``--length``
     metres along its longest horizontal side (a table, a bench) -- centred,
     standing on z = 0 (generator scale is arbitrary);
  3. welds, decimates a copy to ``--tris`` triangles, unwraps it, and bakes the
     high-resolution colour onto it (one material, named after the prop);
  4. exports the .glb plus its albedo .png and renders review pictures
     (front34, front, side, top).

The CRT televisions use their own script (``art/props/tv/scripts/
tv_gameready.py``), which adds the screen glass; this one is that script
without the screen.
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_glb, prev_dir = Path(argv[0]), Path(argv[1]).resolve(), Path(argv[2]).resolve()
opts = {"--height": None, "--length": None, "--front": "-y", "--tris": 4000, "--tex": 1024}
for k, v in zip(argv[3::2], argv[4::2]):
    opts[k] = v if k == "--front" else float(v)
if (opts["--height"] is None) == (opts["--length"] is None):
    sys.exit("give exactly one of --height / --length")
FRONT, TRIS, TEX = opts["--front"], int(opts["--tris"]), int(opts["--tex"])
out_glb.parent.mkdir(parents=True, exist_ok=True)
prev_dir.mkdir(parents=True, exist_ok=True)
name = out_glb.stem

# ------------------------------------------------------------ 1-2. import
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=str(src))
meshes = [o for o in scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
hi = bpy.context.view_layer.objects.active
hi.name = "hi"
for o in list(scene.objects):
    if o is not hi:
        bpy.data.objects.remove(o)
hi.parent = None
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

hi.rotation_mode = "XYZ"                 # glTF imports in quaternion mode
hi.rotation_euler = (0.0, 0.0, {"-y": 0.0, "+x": -math.pi / 2,
                                "+y": math.pi, "-x": math.pi / 2}[FRONT])
bpy.ops.object.transform_apply(rotation=True)


def bbox(o):
    vs = np.array([o.matrix_world @ v.co for v in o.data.vertices])
    return vs.min(0), vs.max(0)


lo_v, hi_v = bbox(hi)
size = hi_v - lo_v
s = (opts["--height"] / size[2] if opts["--height"] is not None
     else opts["--length"] / max(size[0], size[1]))
hi.scale = (s, s, s)
bpy.ops.object.transform_apply(scale=True)
lo_v, hi_v = bbox(hi)
hi.location = (-(lo_v[0] + hi_v[0]) / 2, -(lo_v[1] + hi_v[1]) / 2, -lo_v[2])
bpy.ops.object.transform_apply(location=True)
lo_v, hi_v = bbox(hi)
size = hi_v - lo_v
print(f"SIZE {name}: w={size[0]:.3f} d={size[1]:.3f} h={size[2]:.3f} m")

# ------------------------------------------ 3. weld, decimate, bake
bpy.ops.object.select_all(action="DESELECT")
hi.select_set(True)
bpy.context.view_layer.objects.active = hi
bpy.ops.object.duplicate()
lo = bpy.context.view_layer.objects.active
lo.name = name
n_hi = sum(len(p.vertices) - 2 for p in hi.data.polygons)
# The generator splits vertices along its UV seams; a collapse decimate cannot
# cross a split, so at a 1:350 ratio the shell tears into shards. Weld first.
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=1e-4)
bpy.ops.object.mode_set(mode="OBJECT")
dec = lo.modifiers.new("dec", "DECIMATE")
dec.ratio = min(1.0, TRIS / n_hi)
bpy.ops.object.modifier_apply(modifier="dec")
lo.data.materials.clear()
while lo.data.uv_layers:
    lo.data.uv_layers.remove(lo.data.uv_layers[0])
lo.data.uv_layers.new(name="UVMap")
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004)
bpy.ops.object.mode_set(mode="OBJECT")
n_lo = sum(len(p.vertices) - 2 for p in lo.data.polygons)

img = bpy.data.images.new(f"{name}_albedo", TEX, TEX)
mat = bpy.data.materials.new(name)
mat.use_nodes = True
nt = mat.node_tree
tex_node = nt.nodes.new("ShaderNodeTexImage")
tex_node.image = img
nt.links.new(tex_node.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
nt.nodes.active = tex_node
lo.data.materials.append(mat)

scene.render.engine = "CYCLES"
scene.cycles.samples = 4
scene.cycles.device = "CPU"
bake = scene.render.bake
bake.use_selected_to_active = True
diag = float(np.linalg.norm(size))
bake.cage_extrusion = 0.006 * diag
bake.max_ray_distance = 0.03 * diag
bake.use_pass_direct = False
bake.use_pass_indirect = False
bake.use_pass_color = True
bake.margin = 8
bpy.ops.object.select_all(action="DESELECT")
hi.select_set(True)
lo.select_set(True)
bpy.context.view_layer.objects.active = lo
bpy.ops.object.bake(type="DIFFUSE")
albedo_png = out_glb.parent / f"{name}_albedo.png"
img.filepath_raw = str(albedo_png)
img.file_format = "PNG"
img.save()
print(f"TRIS {name}: hi={n_hi} lo={n_lo}")

# ------------------------------------------------------------ 4. export
bpy.data.objects.remove(hi)
bpy.ops.object.select_all(action="DESELECT")
lo.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(out_glb), use_selection=True,
                          export_format="GLB", export_yup=True)
print(f"WROTE {out_glb}")

scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = scene.render.resolution_y = 800
scene.view_settings.view_transform = "AgX"
world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.57, 0.6, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
scene.world = world
for rot, e in (((50, 0, 30), 3.0), ((60, 0, 210), 1.0)):
    ld = bpy.data.lights.new("sun", "SUN")
    ld.energy = e
    lob = bpy.data.objects.new("sun", ld)
    lob.rotation_euler = [math.radians(a) for a in rot]
    scene.collection.objects.link(lob)
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
span = float(max(size)) * 1.5
cam_data.ortho_scale = span
mid = Vector((0.0, 0.0, size[2] / 2))
views = {
    "front34": Vector((span * 1.1, -span * 1.6, size[2] / 2 + span * 0.8)),
    "front": Vector((0.0, -span * 2, size[2] / 2)),
    "side": Vector((span * 2, 0.0, size[2] / 2)),
    "top": Vector((0.0, -0.001, span * 3)),
}
for v, pos in views.items():
    cam.location = pos
    cam.rotation_euler = (mid - pos).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(prev_dir / f"{name}_{v}.png")
    bpy.ops.render.render(write_still=True)
