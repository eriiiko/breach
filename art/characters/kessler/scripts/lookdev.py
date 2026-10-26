"""Split visor + helmet lamps out of the single-material AI mesh, restyle them, render.
blender -b -P variants.py -- <in.glb> <out_dir>"""
import bpy, bmesh, sys, math, os
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kessler_source as ks  # the visor / lamp selections and the visor cap live there (the game step uses them)

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
obj = next(o for o in bpy.data.objects if o.type == "MESH")
me = obj.data
base_mat = me.materials[0]

fs = ks.face_samples(obj)
cent, nrm, col, rough, lum = fs["cent"], fs["nrm"], fs["col"], fs["rough"], fs["lum"]
nf = len(me.polygons)

P = dict(arg.split("=") for arg in argv[2:])
f = lambda k, d: float(P.get(k, d))
# visor window (Blender Z-up, front faces -Y) and the lamps: kessler_source's selections
visor = ks.visor_mask(fs, vz0=f("vz0", 0.62), vz1=f("vz1", 0.78), vx=f("vx", 0.11), vlum=f("vlum", 0.16), vrmax=f("vrmax", 0.3))
lamp = ks.lamp_mask(fs, llum=f("llum", 0.15))
print("VISOR faces", visor.sum(), "LAMP faces", lamp.sum())
for nm, m in (("visor", visor), ("lamp", lamp)):
    if m.any():
        c = cent[m]; print(nm, "x", c[:, 0].min(), c[:, 0].max(), "y", c[:, 1].min(), c[:, 1].max(), "z", c[:, 2].min(), c[:, 2].max(),
                           "lum", lum[m].mean(), "rough", rough[m].mean())

mode = P.get("mode", "debug")
if P.get("dump"): np.savez(P["dump"], cent=cent, nrm=nrm, col=col, rough=rough, lum=lum)

def mat_flat(name, rgb, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1); b.inputs["Emission Strength"].default_value = emit
    return m

def mat_glass(name, tint, rough_v):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*tint, 1)
    b.inputs["Metallic"].default_value = float(P.get("vmetal", 1.0))  # 1 = mirror-coated visor, low = tinted glass
    b.inputs["Roughness"].default_value = rough_v
    b.inputs["Coat Weight"].default_value = 1.0; b.inputs["Coat Roughness"].default_value = 0.02
    b.inputs["Coat Tint"].default_value = (*tint, 1)
    return m

def mat_lamp(name, rgb, strength):
    # keep the original texture, add emission driven by its own albedo (so the lens pattern glows)
    m = base_mat.copy(); m.name = name
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    tex = next(n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image.name == "Image_0")
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    nt.links.new(tex.outputs["Color"], mix.inputs[6]); mix.inputs[7].default_value = (*rgb, 1)
    nt.links.new(mix.outputs[2], b.inputs["Emission Color"]); b.inputs["Emission Strength"].default_value = strength
    return m

def assign(mask, mat):
    me.materials.append(mat); idx = len(me.materials) - 1
    mi = np.zeros(nf, np.int64); me.polygons.foreach_get("material_index", mi)
    mi[mask] = idx; me.polygons.foreach_set("material_index", mi)

def smooth_visor(mask):
    """Fit a sphere to the visor's vertices and project them onto it: kills the lumpy surface."""
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    vs = {v for i in np.flatnonzero(mask) for v in bm.faces[i].verts}
    # interior verts only: keep the rim welded to the helmet
    inner = [v for v in vs if all(mask[f.index] for f in v.link_faces)]
    X = np.array([tuple(v.co) for v in vs])
    A = np.c_[2 * X, np.ones(len(X))]; bvec = (X ** 2).sum(1)
    sol = np.linalg.lstsq(A, bvec, rcond=None)[0]; c = sol[:3]; r = math.sqrt(sol[3] + c @ c)
    print("sphere fit centre", c, "radius", r, "verts", len(vs), "inner", len(inner))
    for _ in range(int(P.get("relax", 3))):  # a little Laplacian relax, then project
        new = {}
        for v in inner:
            nb = [e.other_vert(v).co for e in v.link_edges]
            new[v] = sum(nb, Vector()) / len(nb)
        for v, p in new.items(): v.co = p
    for v in inner:
        d = Vector(v.co) - Vector(c); v.co = Vector(c) + d.normalized() * r
    for i in np.flatnonzero(mask): bm.faces[i].smooth = True
    bm.to_mesh(me); bm.free()

def visor_cap(mask, mat, push):
    """A new, mathematically smooth visor (`kessler_source.visor_cap`)."""
    co, fit = ks.visor_cap(cent[mask], mat, push=push, half_axes=(f("capx", 0.11), f("capz", 0.075)), name="visor_glass")
    print("cap sphere centre", np.round(fit["centre"], 3), "radius", round(fit["radius"], 3))
    return co

if mode == "debug":
    assign(visor, mat_flat("dbg_visor", (0.0, 1.0, 0.2), 2.0)); assign(lamp, mat_flat("dbg_lamp", (1, 0, 1), 4.0))
else:
    tint = tuple(float(x) for x in P.get("tint", "0.9,0.55,0.15").split(","))
    if P.get("cap", "1") == "1":
        visor_cap(visor, mat_glass("visor_glass", tint, f("vrough", 0.04)), f("push", 0.004))
    else:
        assign(visor, mat_glass("visor_glass", tint, f("vrough", 0.04)))
    assign(lamp, mat_lamp("lamp_on", (1.0, 0.85, 0.6), f("lamp", 25)))

# --- scene
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = sc.render.resolution_y = int(P.get("res", 1024))
sc.view_settings.view_transform = "AgX"
dark = P.get("dark", "0") == "1"
sc.world = bpy.data.worlds.new("w"); sc.world.use_nodes = True
bg = sc.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.35, 0.35, 0.37, 1); bg.inputs[1].default_value = 0.08 if dark else 0.6
if P.get("sky", "1") == "1":
    nt = sc.world.node_tree; sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA" if hasattr(sky, "sun_elevation") else sky.sky_type
    try: sky.sun_elevation = math.radians(35); sky.sun_rotation = math.radians(200)
    except Exception: pass
    nt.links.new(sky.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.01 if dark else 0.12

def sun(name, rot, e):
    l = bpy.data.lights.new(name, "SUN"); l.energy = e
    o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); o.rotation_euler = [math.radians(a) for a in rot]
sun("key", (50, 0, 30), 0.4 if dark else 4.0); sun("rim", (60, 0, 200), 0.3 if dark else 2.5)
if mode != "debug" and P.get("beams", "1") == "1":
    # real light out of each lamp: a spot along -Y
    for x in (-0.165, 0.17):
        l = bpy.data.lights.new("beam", "SPOT"); l.energy = 60; l.spot_size = math.radians(40); l.spot_blend = 0.6
        l.color = (1.0, 0.85, 0.6); l.shadow_soft_size = 0.02
        o = bpy.data.objects.new("beam", l); sc.collection.objects.link(o)
        o.location = (x, -0.25, 0.77); o.rotation_euler = (math.radians(95), 0, 0)
    # a floor + back wall to catch the beams
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.858))
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, -1.6, 0), rotation=(math.radians(90), 0, 0))

cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam); sc.camera = cam
def shot(name, az, el, target, dist, lens=50):
    cam_d.lens = lens; az, el = math.radians(az), math.radians(el)
    pos = target + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * dist
    cam.location = pos; cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(out, name + ".png"); bpy.ops.render.render(write_still=True)
tag = P.get("tag", mode)
for s in P.get("shots", "helmet,three_quarter").split(","):
    if s == "helmet": shot(f"{tag}_helmet", 10, 5, Vector((0, 0, 0.70)), 0.95)
    if s == "helmet34": shot(f"{tag}_helmet34", 40, 10, Vector((0, 0, 0.70)), 0.95)
    if s == "three_quarter": shot(f"{tag}_34", 35, 12, Vector((0, 0, 0)), 3.8)
    if s == "side": shot(f"{tag}_side", 75, 15, Vector((0, -0.6, 0.1)), 3.0, 35)
if P.get("save"):
    bpy.ops.export_scene.gltf(filepath=P["save"], export_format="GLB", use_selection=False)
