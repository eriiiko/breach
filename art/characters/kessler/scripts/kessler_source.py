"""Kessler's high-resolution source for the game-ready step: the AI model imported, rescaled, its
visor replaced and its mesh split into NAMED regions (Blender 4.5).

`build(materials=None)` is the "build" callable `charkit/gameready.py` and `charkit/rig.py` call:
it imports `source/kessler_source.glb` (one mesh, one material, three 2048 px textures), puts it
in the character frame every kit stage uses (faces -Y, +X is his LEFT, feet on z = 0, HEIGHT_M
tall), adds the smooth green visor cap (`visor_cap`) and splits the single mesh into one object
per region (`REGIONS`). The kit then sees what it sees for a scripted character: a set of named
parts whose materials carry the look (the AI material's own base-colour and roughness maps; a
Principled glass on the visor), so gameready bakes colour and gloss from them unchanged, and
`rig.label_skin` / the GEAR table work by part name.

The visor and lamp face selections are the look-dev ones (`scripts/lookdev.py` imports them from
here). They are made in the SOURCE frame (the glb as imported: 1.71 m, centred on the origin),
which is where their thresholds were measured.
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_GLB = os.path.normpath(os.path.join(HERE, "..", "source", "kessler_source.glb"))

# Kessler's game height (Erik may veto: the space marine is ~1.88 m; looks are drawn at true
# relative heights). The source is 1.714 m; every length below is in the scaled frame unless it
# says SOURCE frame.
HEIGHT_M = 1.85

# The visor: deep green glass (Erik, 2026-10-26: option 6 of previews/lookdev/visor_options.png).
VISOR_TINT = (0.02, 0.22, 0.07)
VISOR_ROUGHNESS = 0.04
VISOR_PUSH_M = 0.004  # the cap sits this far outside the fitted sphere (SOURCE frame)
VISOR_HALF_AXES = (0.11, 0.075)  # x / z half-axes of the cap's elliptical cut (SOURCE frame)


# ------------------------------------------------------------------ source frame
def import_source(path=SOURCE_GLB):
    """The glb imported into the current scene; returns its one mesh object (SOURCE frame)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    ob = next(o for o in new if o.type == "MESH")
    for o in new:
        if o is not ob:
            bpy.data.objects.remove(o)
    ob.name = "Kessler_src"
    return ob


def _img(name):
    im = bpy.data.images[name]
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(h, w, 4)


def face_samples(ob):
    """Per face, world frame of `ob`: centroid, normal, albedo RGB and roughness (the mean of three
    corners' texels -- the faces are small), luminance."""
    me = ob.data
    nf = len(me.polygons)
    cent = np.empty(nf * 3, np.float32)
    me.polygons.foreach_get("center", cent)
    nrm = np.empty(nf * 3, np.float32)
    me.polygons.foreach_get("normal", nrm)
    mw = np.array(ob.matrix_world)
    cent = cent.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    nrm = nrm.reshape(-1, 3) @ mw[:3, :3].T
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    ls = np.empty(nf, np.int64)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(nf, np.int64)
    me.polygons.foreach_get("loop_total", lt)
    alb, mr = _img("Image_0"), _img("Image_1")
    H, W = alb.shape[:2]

    def sample(img, k):
        cols = []
        for j in range(3):
            u = uv[ls + np.minimum(j, lt - 1)]
            cols.append(img[np.clip((u[:, 1] * H).astype(int), 0, H - 1), np.clip((u[:, 0] * W).astype(int), 0, W - 1), :k])
        return np.mean(cols, axis=0)

    col = sample(alb, 3)
    rough = sample(mr, 3)[:, 1]
    lum = col @ np.array([0.2126, 0.7152, 0.0722])
    return dict(cent=cent, nrm=nrm, col=col, rough=rough, lum=lum)


def visor_mask(fs, vz0=0.62, vz1=0.78, vx=0.11, vlum=0.16, vrmax=0.3):
    """The model's own visor faces (SOURCE frame): the dark glossy window on the helmet's front."""
    c, n = fs["cent"], fs["nrm"]
    return ((np.abs(c[:, 0]) < vx) & (c[:, 2] > vz0) & (c[:, 2] < vz1) & (n[:, 1] < -0.25)
            & (fs["lum"] < vlum) & (fs["rough"] < vrmax) & (c[:, 1] < -0.05))


def lamp_mask(fs, llum=0.15):
    """The two helmet lamps (SOURCE frame): bright warm faces on each side of the helmet's front.
    Kept as a selection for the flashlight feature; nothing makes them glow yet."""
    c, n = fs["cent"], fs["nrm"]
    ax = np.abs(c[:, 0])
    return ((ax > 0.14) & (ax < 0.21) & (c[:, 2] > 0.74) & (c[:, 2] < 0.795) & (c[:, 1] < -0.10)
            & (n[:, 1] < -0.5) & (fs["lum"] > llum))


def visor_cap(X, mat, push=VISOR_PUSH_M, half_axes=VISOR_HALF_AXES, n=48, name="Visor"):
    """A new, mathematically smooth visor: a sphere fitted (least squares) to the points X (the old
    visor's face centroids), grown by `push`, cut to an x/z ellipse round their middle. Returns
    the new object (same frame as X), its one material `mat`, and the fit."""
    A = np.c_[2 * X, np.ones(len(X))]
    sol = np.linalg.lstsq(A, (X ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    r = math.sqrt(sol[3] + c @ c) + push
    x0, z0 = 0.5 * (X[:, 0].min() + X[:, 0].max()), 0.5 * (X[:, 2].min() + X[:, 2].max())
    ex, ez = half_axes
    bm = bmesh.new()
    grid = {}
    for i in range(n + 1):
        for j in range(n + 1):
            u, v = -1 + 2 * i / n, -1 + 2 * j / n
            if u * u + v * v > 1.0001:
                continue
            x, z = x0 + u * ex, z0 + v * ez
            dy2 = r * r - (x - c[0]) ** 2 - (z - c[2]) ** 2
            grid[i, j] = bm.verts.new((x, c[1] - math.sqrt(max(dy2, 0)), z))
    for i in range(n):
        for j in range(n):
            q = [grid.get(k) for k in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
            if all(q):
                bm.faces.new(q).smooth = True  # wound so the normals face OUT (-y): an inward cap bakes black AO
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    # a planar UV so the mesh carries a UV layer like every other part (the bake reads none of it)
    uvl = me.uv_layers.new(name="UVMap")
    co = np.array([v.co[:] for v in me.vertices])
    for lp in me.loops:
        p = co[lp.vertex_index]
        uvl.data[lp.index].uv = ((p[0] - x0) / (2 * ex) + 0.5, (p[2] - z0) / (2 * ez) + 0.5)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(mat)
    return ob, dict(centre=c.tolist(), radius=r)


def mat_visor_glass(tint=VISOR_TINT, rough=VISOR_ROUGHNESS, metallic=1.0):
    """The look-dev glass: tinted, mirror-coated, clear-coated. For the game bake only its Base
    Color (the tint) and Roughness / Coat (gloss ~1) count."""
    m = bpy.data.materials.new("visor_glass")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*tint, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.02
    b.inputs["Coat Tint"].default_value = (*tint, 1)
    return m


def game_frame_matrix(ob):
    """SOURCE frame -> character frame: uniform scale to HEIGHT_M, soles on z = 0 (x, y kept)."""
    co = np.empty(len(ob.data.vertices) * 3, np.float32)
    ob.data.vertices.foreach_get("co", co)
    z = co.reshape(-1, 3)[:, 2]
    s = HEIGHT_M / float(z.max() - z.min())
    return Matrix.Translation((0.0, 0.0, -float(z.min()) * s)) @ Matrix.Scale(s, 4), s


# ---------------------------------------------------------------------- regions
# Each face of the AI mesh goes to the FIRST region whose test its centroid passes (character
# frame, metres; `ax` = |x|, the tests are mirror-symmetric, so one rule covers both sides and the
# part name carries the side: `<Region>_L` where x > 0). Boundaries read off the model's own
# geometry (ortho grids and ray probes of the scaled mesh, 2026-10-26). A region exists to be
# named in rig_spec.GEAR; what no region claims is "Suit" (bone-heat weights).
def _regions(c, ax):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    return (
        ("Glove", ax > 0.845),  # glove + the hand-back plate (x 0.78..0.97 over the fingers starts at the cuff)
        ("Gauntlet", (ax > 0.63) & (z > 1.15)),  # the forearm armour block, x 0.60..0.845
        ("Helmet", (z > 1.565) & (ax < 0.25) & (y < 0.07)),  # helmet shell from its bottom rim (z 1.57) up
        ("Pack_Hose", (ax > 0.27) & (y > 0.12) & (z > 1.35)),  # the hoses from the pack to the arms
        ("Pack", (y > 0.12) & (z > 1.0) & (ax < 0.34)),  # backpack and its tanks (the torso's back is y ~0.12)
        ("Chest", (y < -0.20) & (z > 1.12) & (z < 1.56) & (ax < 0.24)),  # chest plate
        ("Coil", (x < -0.19) & (y < -0.22) & (z > 0.58) & (z < 0.93)),  # the yellow cable coil on his right hip
        ("Belt", (z > 0.92) & (z < 1.07) & (ax < 0.36)),  # belt + its side boxes
        ("Plate", (y < -0.22) & (z > 0.62) & (z <= 0.92) & (ax < 0.17)),  # the front plate hanging from the belt
        # the toe cap: in front of the ball joint (rig_spec: (0.24, -0.185)) along the foot's 15 deg toe-out line
        ("Boot_Toe", (z < 0.12) & ((ax - 0.24) * 0.259 - (y + 0.185) * 0.966 > 0.0)),
        ("Boot", z < 0.075),  # sole and heel
        ("Boot_Shaft", z < 0.20),  # the boot above the ankle
        ("Shin_Plate", (y < -0.13) & (z < 0.585) & (ax > 0.05) & (ax < 0.36)),  # the big knee + shin plate
    )


def region_labels(cent):
    """Per face (character frame centroids): its part name ("Suit" when no region claims it)."""
    ax = np.abs(cent[:, 0])
    lab = np.full(len(cent), "Suit", dtype=object)
    free = np.ones(len(cent), bool)
    for name, m in _regions(cent, ax):
        hit = free & m
        lab[hit] = name
        free &= ~hit
    side = np.where(cent[:, 0] > 0, "_L", "_R")
    return np.array([n + s for n, s in zip(lab, side)], dtype=object)


def _split(ob, labels):
    """`ob` split into one object per distinct label (named by it); keeps every attribute, the UVs
    and the custom normals (Blender's own separate-by-selection). The label travels with each face
    as a face attribute, so the split never depends on face order."""
    names = sorted(set(labels.tolist()))
    code = {n: i for i, n in enumerate(names)}
    at = ob.data.attributes.new("_region", "INT", "FACE")
    at.data.foreach_set("value", np.array([code[n] for n in labels], np.int32))
    for o in bpy.context.scene.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    out = []
    for nm in names[:-1]:
        before = set(bpy.data.objects)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bm = bmesh.from_edit_mesh(ob.data)
        lay = bm.faces.layers.int["_region"]
        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        for f in bm.faces:
            f.select = f[lay] == code[nm]
        bm.select_flush_mode()
        bmesh.update_edit_mesh(ob.data)
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        new = next(o for o in bpy.data.objects if o not in before)
        new.name = new.data.name = nm
        new.select_set(False)
        out.append(new)
    ob.name = ob.data.name = names[-1]
    out.append(ob)
    for o in out:
        o.data.attributes.remove(o.data.attributes["_region"])
    return out


def build(materials=None, path=SOURCE_GLB, log=None):
    """The kit's build callable (see the module doc). `materials` is unused: the look is the AI
    model's own material plus the visor glass. Returns the region objects (character frame)."""
    ob = import_source(path)
    fs = face_samples(ob)
    vmask, lmask = visor_mask(fs), lamp_mask(fs)
    cap, fit = visor_cap(fs["cent"][vmask], mat_visor_glass())
    M, s = game_frame_matrix(ob)
    for o in (ob, cap):
        o.data.transform(M)
        o.matrix_world = Matrix.Identity(4)
    cent = (np.c_[fs["cent"], np.ones(len(fs["cent"]))] @ np.array(M).T)[:, :3]
    labels = region_labels(cent)
    side = np.where(cent[:, 0] > 0, "_L", "_R")
    labels[lmask] = np.array(["Lamp" + x for x in side[lmask]], dtype=object)
    labels[vmask] = np.array(["Visor_Old" + x for x in side[vmask]], dtype=object)  # under the cap
    parts = _split(ob, labels)
    if log is not None:
        log.update(scale=s, visor_faces=int(vmask.sum()), lamp_faces=int(lmask.sum()), visor_fit=fit,
                   regions={p.name: len(p.data.polygons) for p in parts})
    return parts + [cap]
