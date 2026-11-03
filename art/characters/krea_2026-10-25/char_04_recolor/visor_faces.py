"""Dump the UV triangles of the helmet's front faces (head height, facing -Y)
so the visor can be masked in the texture.   blender -b -P visor_faces.py -- in.glb out.json"""
import bpy, sys, json
a = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
o = [x for x in bpy.context.scene.objects if x.type == "MESH"][0]
me = o.data
mw = o.matrix_world
zs = [(mw @ v.co).z for v in me.vertices]
zmin, zmax = min(zs), max(zs)
uv = me.uv_layers.active.data
me.calc_loop_triangles()
out = []
for t in me.loop_triangles:
    c = sum(((mw @ me.vertices[i].co) for i in t.vertices), start=bpy.app.driver_namespace.get("_", None) or type(mw.col[0].xyz)((0, 0, 0))) / 3
    h = (c.z - zmin) / (zmax - zmin)
    n = (mw.to_3x3() @ t.normal).normalized()
    if h > 0.80:
        out.append({"h": h, "ny": n.y, "x": c.x, "uv": [list(uv[l].uv) for l in t.loops]})
json.dump({"zmin": zmin, "zmax": zmax, "tris": out}, open(a[1], "w"))
print("HEADTRIS", len(out))
