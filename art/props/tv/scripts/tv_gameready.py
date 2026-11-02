"""Game-ready step for a generated CRT television (Krea .glb -> game .glb).

    blender -b -P tv_gameready.py -- <in.glb> <out.glb> <preview_dir>
        [--width 0.50] [--front -y] [--tris 4000] [--bulge 0.04] [--seed U V]
        [--rect X0 X1 Z0 Z1] [--round 5]

What it does, in order:
  1. imports the generator's single-mesh .glb and turns it so its screen faces
     -Y (``--front`` names the axis the screen faces in the source);
  2. scales it to ``--width`` metres across, centred, standing on z = 0
     (generator scale is arbitrary);
  3. finds the screen: on a front-on depth map, the region around the seed
     (default: the centre) lying proud of the recessed ring between a bulging
     CRT face and its bezel; its bounding box is the screen opening. That
     ring is not always closed, so ``--rect`` (metres, after step 2: x across,
     z up) overrides it -- read it off ``<name>_flatfront.png`` -- and the
     mask picture shows what was used;
  4. builds OUR screen glass over it: a rounded-rectangle (superellipse) patch
     laid on the old screen's surface and bulged a further ``--bulge`` x its
     height at the centre, with its
     own material ``tv_screen`` and UVs 0..1 over the whole glass, so a
     picture can be shown on it edge to edge (dark glass = switched off);
  5. decimates a copy of the body to ``--tris`` triangles, unwraps it, and
     bakes the high-resolution colour onto it (material ``tv_body``);
  6. exports body + glass as one .glb and renders review pictures, including
     the glass showing a test pattern (the "on" proof).

No generator splits the screen out (Krea TV test, 2026-11-02), which is why
step 4 exists.
"""
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

# ---------------------------------------------------------------- arguments
argv = sys.argv[sys.argv.index("--") + 1:]
src, out_glb, prev_dir = Path(argv[0]), Path(argv[1]).resolve(), Path(argv[2]).resolve()
opts = {"--width": 0.50, "--front": "-y", "--tris": 4000, "--bulge": 0.04,
        "--seed": None, "--rect": None, "--round": 5.0}
i = 3
while i < len(argv):
    k = argv[i]
    if k == "--seed":
        opts[k] = (float(argv[i + 1]), float(argv[i + 2]))
        i += 3
        continue
    if k == "--rect":
        opts[k] = tuple(float(a) for a in argv[i + 1:i + 5])
        i += 5
        continue
    opts[k] = argv[i + 1] if k == "--front" else float(argv[i + 1])
    i += 2
WIDTH, FRONT, TRIS, BULGE = opts["--width"], opts["--front"], int(opts["--tris"]), opts["--bulge"]
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
for o in list(scene.objects):          # drop import empties, keep the mesh
    if o is not hi:
        bpy.data.objects.remove(o)
hi.parent = None
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

turn = {"-y": 0.0, "+x": math.pi / 2, "+y": math.pi, "-x": -math.pi / 2}[FRONT]
hi.rotation_mode = "XYZ"                 # glTF imports in quaternion mode
hi.rotation_euler = (0.0, 0.0, -turn if FRONT in ("+x", "-x") else turn)
bpy.ops.object.transform_apply(rotation=True)


def bbox(o):
    vs = np.array([o.matrix_world @ v.co for v in o.data.vertices])
    return vs.min(0), vs.max(0)


lo_v, hi_v = bbox(hi)
s = WIDTH / (hi_v[0] - lo_v[0])
hi.scale = (s, s, s)
bpy.ops.object.transform_apply(scale=True)
lo_v, hi_v = bbox(hi)
hi.location = (-(lo_v[0] + hi_v[0]) / 2, -(lo_v[1] + hi_v[1]) / 2, -lo_v[2])
bpy.ops.object.transform_apply(location=True)
lo_v, hi_v = bbox(hi)
size = hi_v - lo_v
print(f"SIZE {name}: w={size[0]:.3f} d={size[1]:.3f} h={size[2]:.3f} m")

# ------------------------------------------------- 3. find the screen
RES = 512
ortho = max(size[0], size[2]) * 1.1
cam_z = size[2] / 2
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam


def aim(pos, target=(0.0, 0.0, 0.0), scale=None):
    cam.location = Vector(pos)
    cam.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    if scale:
        cam_data.ortho_scale = scale


scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "FLAT"
scene.display.shading.color_type = "TEXTURE"
scene.render.resolution_x = scene.render.resolution_y = RES
scene.view_settings.view_transform = "Standard"
aim((0.0, -10.0, cam_z), (0.0, 0.0, cam_z), ortho)
flat_png = prev_dir / f"{name}_flatfront.png"
scene.render.filepath = str(flat_png)
bpy.ops.render.render(write_still=True)

img = bpy.data.images.load(str(flat_png))
px = np.array(img.pixels[:]).reshape(RES, RES, 4)[..., :3]   # row 0 = bottom
seed = opts["--seed"] or (0.5, 0.5)
sr, sc = int(seed[1] * (RES - 1)), int(seed[0] * (RES - 1))
# The screen is found by DEPTH, not colour: its painted reflections split it
# into bands a colour fill stops at, but it is always recessed behind the
# bezel. A depth map from front-on rays; the screen is the region around the
# seed lying deeper than halfway between the cabinet front and the seed.
dg = bpy.context.evaluated_depsgraph_get()
D = 128
depth = np.full((D, D), np.inf)
for r in range(D):
    for c in range(D):
        x = ((c + 0.5) / D - 0.5) * ortho
        z = cam_z + ((r + 0.5) / D - 0.5) * ortho
        hit, loc, *_ = scene.ray_cast(dg, Vector((x, -10.0, z)), Vector((0, 1, 0)))
        if hit:
            depth[r, c] = loc.y
dr, dc = int(seed[1] * (D - 1)), int(seed[0] * (D - 1))
y_front = depth[np.isfinite(depth)].min()



def ring_depth(line):
    """Deepest point walking out from the seed before the surface rises."""
    best = line[0]
    for v in line[1:]:
        if not np.isfinite(v) or v < best - 0.002:
            break
        best = max(best, v)
    return best


y0 = depth[dr, dc]
rings = [ring_depth(depth[dr, dc::-1]), ring_depth(depth[dr, dc:]),
         ring_depth(depth[dr::-1, dc]), ring_depth(depth[dr:, dc])]
# A bulging CRT face sits proud of a recessed ring between it and the bezel:
# the screen is everything in front of that ring, bounded by it.
thr = min(rings) - 0.0015
print(f"DEPTH {name}: rings={[round(r, 4) for r in rings]} thr={thr:.4f}")
ok_d = np.isfinite(depth) & (depth < thr)
ok = np.kron(ok_d, np.ones((RES // D, RES // D), bool))
fill = np.zeros_like(ok)
fill[sr, sc] = True
while True:                              # connected component by dilation
    grow = fill.copy()
    grow[1:] |= fill[:-1]
    grow[:-1] |= fill[1:]
    grow[:, 1:] |= fill[:, :-1]
    grow[:, :-1] |= fill[:, 1:]
    grow &= ok
    if (grow == fill).all():
        break
    fill = grow
if not fill.any():
    # A flat or recessed face has no proud region at the seed: nothing to
    # find. Only --rect can say where the screen is; the mask shows the seed.
    print(f"WARNING {name}: no screen found at the seed; pass --rect")
    fill[max(sr - 4, 0):sr + 5, max(sc - 4, 0):sc + 5] = True
rows, cols = np.nonzero(fill)
# Trim 2 % per side against stray reflections reaching into the bezel.
r0, r1 = np.percentile(rows, [2, 98])
c0, c1 = np.percentile(cols, [2, 98])
to_x = lambda c: (c / RES - 0.5) * ortho
to_z = lambda r: cam_z + (r / RES - 0.5) * ortho
sx0, sx1, sz0, sz1 = opts["--rect"] or (to_x(c0), to_x(c1), to_z(r0), to_z(r1))
print(f"SCREEN {name}: x[{sx0:.3f},{sx1:.3f}] z[{sz0:.3f},{sz1:.3f}] "
      f"({fill.sum()} px filled of {RES * RES})")
mask_png = prev_dir / f"{name}_screenmask.png"
vis = np.dstack([px * 0.5 + fill[..., None] * np.array([0.5, 0.0, 0.0]),
                 np.ones((RES, RES, 1))]).astype(np.float32)
mimg = bpy.data.images.new("mask", RES, RES)
mimg.pixels[:] = vis.ravel()
mimg.filepath_raw = str(mask_png)
mimg.file_format = "PNG"
mimg.save()

# ------------------------------------------------- 4. the screen glass
NU, NV, P = 48, 36, opts["--round"]
cx, cz = (sx0 + sx1) / 2, (sz0 + sz1) / 2
hx, hz = (sx1 - sx0) / 2, (sz1 - sz0) / 2
dg = bpy.context.evaluated_depsgraph_get()


def ray_y(x, z):
    hit, loc, *_ = scene.ray_cast(dg, Vector((x, -10.0, z)), Vector((0, 1, 0)))
    return loc.y if hit else None


# The glass lies on the old screen surface (ray-cast per vertex, so its rim
# meets the bezel instead of floating in front of it) plus an extra bulge.
bulge_m = BULGE * (2 * hz)
y_fallback = ray_y(cx, cz)

bm = bmesh.new()
uv_layer = bm.loops.layers.uv.new("UVMap")
grid = []
for j in range(NV + 1):
    row = []
    v = -1 + 2 * j / NV
    for k in range(NU + 1):
        u = -1 + 2 * k / NU
        r = (abs(u) ** P + abs(v) ** P) ** (1 / P)
        m = max(abs(u), abs(v)) / r if r > 0 else 0.0   # square -> superellipse
        x, z = cx + hx * u * m, cz + hz * v * m
        y_hit = ray_y(x, z)
        y = (y_fallback if y_hit is None else y_hit) - 0.0015 \
            - bulge_m * (1 - u * u) * (1 - v * v)
        row.append((bm.verts.new((x, y, z)), (k / NU, j / NV)))
    grid.append(row)
for j in range(NV):
    for k in range(NU):
        quad = [grid[j][k], grid[j][k + 1], grid[j + 1][k + 1], grid[j + 1][k]]
        f = bm.faces.new([q[0] for q in quad])
        for loop, q in zip(f.loops, quad):
            loop[uv_layer].uv = q[1]
bm.normal_update()
glass_mesh = bpy.data.meshes.new("tv_screen")
bm.to_mesh(glass_mesh)
bm.free()
glass = bpy.data.objects.new("tv_screen", glass_mesh)
scene.collection.objects.link(glass)
for p in glass_mesh.polygons:
    p.use_smooth = True
if glass_mesh.polygons[0].normal.y > 0:     # face the viewer (-Y)
    glass_mesh.flip_normals()

screen_mat = bpy.data.materials.new("tv_screen")
screen_mat.use_nodes = True
bsdf = screen_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.035, 0.045, 0.042, 1)
bsdf.inputs["Roughness"].default_value = 0.12
glass_mesh.materials.append(screen_mat)

# ------------------------------------------- 5. decimate + bake the body
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

TEX = 1024
body_img = bpy.data.images.new(f"{name}_albedo", TEX, TEX)
body_mat = bpy.data.materials.new("tv_body")
body_mat.use_nodes = True
nt = body_mat.node_tree
tex_node = nt.nodes.new("ShaderNodeTexImage")
tex_node.image = body_img
nt.links.new(tex_node.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
nt.nodes.active = tex_node
lo.data.materials.append(body_mat)

scene.render.engine = "CYCLES"
scene.cycles.samples = 4
scene.cycles.device = "CPU"
bake = scene.render.bake
bake.use_selected_to_active = True
bake.cage_extrusion = 0.004
bake.max_ray_distance = 0.02
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
body_img.filepath_raw = str(albedo_png)
body_img.file_format = "PNG"
body_img.save()
print(f"TRIS {name}: hi={n_hi} body={n_lo} glass={len(glass_mesh.polygons) * 2}")

# ------------------------------------------------------------ 6. export
bpy.data.objects.remove(hi)
bpy.ops.object.select_all(action="DESELECT")
lo.select_set(True)
glass.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(out_glb), use_selection=True,
                          export_format="GLB", export_yup=True)
print(f"WROTE {out_glb}")

# ------------------------------------------------------------ previews
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

span = max(size) * 1.5
mid = (0.0, 0.0, size[2] / 2)
views = {
    "front34": ((span * 1.1, -span * 1.6, size[2] / 2 + span * 0.8), mid),
    "side": ((span * 2, 0.0, size[2] / 2), mid),
    "back": ((-span * 0.8, span * 2, size[2] / 2 + span * 0.6), mid),
    "top": ((0.0, -0.001, span * 3), mid),
    "front": ((0.0, -span * 2, size[2] / 2), mid),
}


def shoot(tag):
    for v, (pos, tgt) in views.items():
        aim(pos, tgt, span)
        scene.render.filepath = str(prev_dir / f"{name}_{tag}_{v}.png")
        bpy.ops.render.render(write_still=True)


shoot("off")

# "On": a test pattern on the glass proves the UVs (corners, orientation).
T = 256
yy, xx = np.mgrid[0:T, 0:T] / (T - 1)
pat = np.zeros((T, T, 4), np.float32)
check = ((xx * 8).astype(int) + (yy * 6).astype(int)) % 2
pat[..., 0] = 0.2 + 0.6 * xx
pat[..., 1] = 0.2 + 0.6 * yy
pat[..., 2] = 0.3 + 0.4 * check
pat[..., 3] = 1
pat[(yy > 0.75) & (abs(xx - 0.5) < (0.95 - yy))] = (1, 1, 0.2, 1)   # "up" arrow
pimg = bpy.data.images.new("pattern", T, T)
pimg.pixels[:] = pat.ravel()
pn = screen_mat.node_tree.nodes.new("ShaderNodeTexImage")
pn.image = pimg
sb = screen_mat.node_tree.nodes["Principled BSDF"]
screen_mat.node_tree.links.new(pn.outputs["Color"], sb.inputs["Emission Color"])
sb.inputs["Emission Strength"].default_value = 2.0
shoot("on")
