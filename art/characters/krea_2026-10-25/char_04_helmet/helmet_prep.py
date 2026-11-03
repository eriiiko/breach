"""Tripo beak helmet -> game mesh: weld, decimate, keep its UVs and textures,
save the base-colour texture as PNG for recolouring.
    blender -b -P helmet_prep.py -- <in.glb> <out.glb> <basecolor_out.png> [--tris 40000]
"""
import sys
import bpy
a = sys.argv[sys.argv.index("--") + 1:]
src, out, tex_out = a[:3]
tris = int(a[a.index("--tris") + 1]) if "--tris" in a else 40000
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
print("MESHES", [(o.name, len(o.data.polygons), [m.name for m in o.data.materials]) for o in meshes])
ob = meshes[0]
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
vs = [v.co for v in ob.data.vertices]
for i, c in enumerate("xyz"):
    print(f"BBOX {c}: {min(v[i] for v in vs):.3f} .. {max(v[i] for v in vs):.3f}")
for n in ob.active_material.node_tree.nodes:
    if n.type == "TEX_IMAGE":
        print("TEX", n.image.name, n.image.size[:], [l.to_socket.name for l in n.outputs[0].links])
bsdf = next(n for n in ob.active_material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
img = bsdf.inputs["Base Color"].links[0].from_node.image
img.filepath_raw = tex_out
img.file_format = "PNG"
img.save()
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=1e-5)
bpy.ops.object.mode_set(mode="OBJECT")
n = len(ob.data.polygons)
m = ob.modifiers.new("dec", "DECIMATE")
m.ratio = min(1.0, tris / max(1, sum(len(p.vertices) - 2 for p in ob.data.polygons)))
bpy.ops.object.modifier_apply(modifier="dec")
print("TRIS", n, "->", len(ob.data.polygons))
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
