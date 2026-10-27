"""Blender 4.5: whole octopus -- editable hero .blend, baked game GLB, review renders.

blender -b --factory-startup -P build_octopus.py -- --root <checkout> [--stage assets|renders|gif|all]

Builds on the one-arm proof: geometry, controls and skinning come from
octopus_math (which imports arm_math); Blender only receives poses. Ordinary
meshes + linear armature modifiers; no constraints, drivers or startup scripts.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_math as arm
import octopus_math as om

HOME = HERE.parent
OUT = HOME / "prototype" / "octopus"
PREVIEW = HOME / "previews" / "octopus"
B = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])     # glTF Y-up -> Blender Z-up
TEX = 2048


def log(*a):
    print("[octopus]", *a, flush=True)


def gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                bpy.context.scene.cycles.device = "GPU"
                log("cycles device", kind, [d.name for d in devs])
                return
        except Exception as exc:  # pragma: no cover
            log("no", kind, exc)


# --------------------------------------------------------------------------
# Materials (hero: procedural, stable under deformation via rest 'Generated' coords)
def node(t, kind, **inputs):
    n = t.nodes.new(kind)
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def skin_material():
    P = arm.PALETTE
    m = bpy.data.materials.new("Octopus_skin")
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    tc = t.nodes.new("ShaderNodeTexCoord")
    mp = t.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.9, 2.9, 1.2)       # Generated is bbox-normalised
    t.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    big = node(t, "ShaderNodeTexNoise", Scale=9.0, Detail=5.0, Roughness=.62)
    t.links.new(mp.outputs["Vector"], big.inputs["Vector"])
    ramp = t.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = .36, .72
    ramp.color_ramp.elements[0].color = (.25, .062, .025, 1)
    ramp.color_ramp.elements[1].color = (.52, .20, .085, 1)
    t.links.new(big.outputs["Fac"], ramp.inputs["Fac"])
    spots = node(t, "ShaderNodeTexVoronoi", Scale=55.0, Randomness=.9)
    t.links.new(mp.outputs["Vector"], spots.inputs["Vector"])
    sp = t.nodes.new("ShaderNodeMapRange")
    sp.inputs["From Min"].default_value, sp.inputs["From Max"].default_value = .0, .16
    sp.inputs["To Min"].default_value, sp.inputs["To Max"].default_value = .38, 0.
    t.links.new(spots.outputs["Distance"], sp.inputs["Value"])
    light = t.nodes.new("ShaderNodeMix")
    light.data_type = "RGBA"
    light.inputs[7].default_value = (.66, .34, .18, 1)
    t.links.new(sp.outputs["Result"], light.inputs[0])
    t.links.new(ramp.outputs["Color"], light.inputs[6])
    attr = t.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "tint"
    sep = t.nodes.new("ShaderNodeSeparateColor")
    t.links.new(attr.outputs["Color"], sep.inputs["Color"])
    pale = t.nodes.new("ShaderNodeMix")
    pale.data_type = "RGBA"
    pale.inputs[7].default_value = (*P["oral"], 1)
    t.links.new(sep.outputs["Red"], pale.inputs[0])
    t.links.new(light.outputs[2], pale.inputs[6])
    mouth = t.nodes.new("ShaderNodeMix")
    mouth.data_type = "RGBA"
    mouth.inputs[7].default_value = (.16, .06, .05, 1)
    t.links.new(sep.outputs["Green"], mouth.inputs[0])
    t.links.new(pale.outputs[2], mouth.inputs[6])
    t.links.new(mouth.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = .44
    bsdf.inputs["Specular IOR Level"].default_value = .45
    bsdf.inputs["Subsurface Weight"].default_value = .06
    bsdf.inputs["Subsurface Radius"].default_value = (.02, .008, .005)
    bump = t.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .08
    bump.inputs["Distance"].default_value = .004
    t.links.new(spots.outputs["Distance"], bump.inputs["Height"])
    t.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    m["game_gloss_note"] = "moist satin: roughness .44 -> game gloss ~.32"
    return m


def cup_material(rim=None, bowl=None, name="Octopus_suckers"):
    """Sucker material; rim/bowl (linear RGB) default to arm.PALETTE (a pattern row passes its own)."""
    P = arm.PALETTE
    rim, bowl = rim or P["rim"], bowl or P["bowl"]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    attr = t.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = "tint"
    sep = t.nodes.new("ShaderNodeSeparateColor")
    t.links.new(attr.outputs["Color"], sep.inputs["Color"])
    mix = t.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*rim, 1)
    mix.inputs[7].default_value = (*bowl, 1)
    t.links.new(sep.outputs["Blue"], mix.inputs[0])
    t.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = .36
    bsdf.inputs["Subsurface Weight"].default_value = .05
    return m


def eye_image():
    n = 256
    y, x = (np.mgrid[0:n, 0:n] + .5) / n
    dx, dy = x - .5, y - .5
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    iris_r = .17
    streak = .5 + .5 * np.sin(ang * 46) * np.sin(ang * 13 + r * 40)
    gold = np.array([.78, .55, .12]) * (.75 + .25 * streak[..., None])
    outer = np.array([.35, .17, .06])
    t = np.clip((r - iris_r * .78) / (iris_r * .22), 0, 1)[..., None]
    col = gold * (1 - t) + np.array([.18, .07, .03]) * t
    skin = np.clip((r - iris_r) / .02, 0, 1)[..., None]
    col = col * (1 - skin) + outer * skin
    # horizontal (dumbbell-ish) slit pupil
    pw, ph = .105, .028 + .012 * (np.abs(dx) / .105)
    pupil = ((np.abs(dx) / pw) ** 2.4 + (np.abs(dy) / ph) ** 2) < 1
    col = np.where(pupil[..., None], np.array([.01, .01, .012]), col)
    rgba = np.concatenate([col, np.ones((n, n, 1))], -1).astype(np.float32)
    img = bpy.data.images.new("Octopus_eye", n, n, alpha=False)
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def eye_material():
    m = bpy.data.materials.new("Octopus_eye")
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    tex = t.nodes.new("ShaderNodeTexImage")
    tex.image = eye_image()
    uv = t.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "Eye_UV"
    t.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    t.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = .07
    bsdf.inputs["Coat Weight"].default_value = .6
    return m


# --------------------------------------------------------------------------
def build_mesh(data, name, materials):
    V = data["vertices"] * om.OCTOPUS_SCALE
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata((V @ B.T).tolist(), [], data["faces"])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    kind = data["kind"]
    mat_of_kind = {0: 0, 1: 1, 2: 2, 3: 0}
    for m in materials:
        mesh.materials.append(m)
    pidx = []
    for p in mesh.polygons:
        pidx.append(mat_of_kind[int(kind[p.vertices[0]])])
    mesh.polygons.foreach_set("material_index", pidx)
    tint = np.zeros((len(V), 4), np.float32)
    tint[:, 0], tint[:, 1], tint[:, 2], tint[:, 3] = data["pale"], data["mouth"], data["bowl"], 1
    ca = mesh.color_attributes.new("tint", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", tint.ravel())
    uv = mesh.uv_layers.new(name="Eye_UV")
    uvs = np.full((len(V), 2), .02)
    for (a, b), e in zip(data["eye_ranges"], data["eye_uv"]):
        uvs[a:b] = np.array(e)
    loops = np.zeros(len(mesh.loops), int)
    mesh.loops.foreach_get("vertex_index", loops)
    uv.data.foreach_set("uv", uvs[loops].ravel())
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def build_rig(data, name, mesh_obj):
    names, bind = data["bones"], data["bind"] * om.OCTOPUS_SCALE
    rd = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, rd)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    lod = data["lod"]
    seg = arm.STEP * (1 if lod == "hero" else om.GAME_EVERY) * om.OCTOPUS_SCALE
    for i, n in enumerate(names):
        b = rd.edit_bones.new(n)
        head = bind[i]
        if "_" in n and n.split("_")[0] in om.ARM_NAMES:
            k = om.ARM_NAMES.index(n.split("_")[0])
            tail = head + om.MOUNT_FRAMES[k][:, 0] * seg
        elif n == "siphon":
            tail = head + om.SIPHON_DIR / np.linalg.norm(om.SIPHON_DIR) * .08 * om.OCTOPUS_SCALE
        elif n.startswith("mantle_squeeze"):
            tail = head - om.WALL_DIR[n] * .05 * om.OCTOPUS_SCALE
        elif n == "mantle":
            tail = (om.MANTLE_C + [0, 0, -.2]) * om.OCTOPUS_SCALE
        else:
            tail = head + np.array([0., .12, 0.]) * om.OCTOPUS_SCALE
        b.head, b.tail = Vector(B @ head), Vector(B @ tail)
        eb[n] = b
    for n in names:
        p = om.bone_parent(n)
        if p:
            eb[n].parent = eb[p]
            eb[n].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
    rig.show_in_front = True
    # vertex groups (vectorised by bone)
    J, W = data["joints"], data["weights"]
    groups = [mesh_obj.vertex_groups.new(name=n) for n in names]
    for bi, g in enumerate(groups):
        rows, cols = np.nonzero((J == bi) & (W > 0))
        if len(rows) == 0:
            continue
        w = W[rows, cols]
        order = np.argsort(w)
        for val in np.unique(np.round(w, 6)):
            sel = rows[np.isclose(w, val, atol=5e-7)]
            g.add(sel.tolist(), float(val), "REPLACE")
    mod = mesh_obj.modifiers.new("Portable_linear_skin", "ARMATURE")
    mod.object = rig
    mod.use_deform_preserve_volume = False
    mesh_obj.parent = rig
    return rig


def set_pose(rig, body, arms, lod, keyframe=None):
    names = om.bones(lod)
    pts, rots = om.pose_octopus(body, arms, lod)
    target = {}
    for i, n in enumerate(names):
        bone = rig.data.bones[n]
        rot = Matrix((B @ rots[i] @ B.T).tolist()) @ bone.matrix_local.to_3x3()
        m = rot.to_4x4()
        m.translation = Vector(B @ pts[i])
        target[n] = m
    for n in names:
        pb = rig.pose.bones[n]
        rest = pb.bone.matrix_local
        p = om.bone_parent(n)
        if p is None:
            basis = rest.inverted() @ target[n]
        else:
            prest = rig.data.bones[p].matrix_local
            basis = (prest.inverted() @ rest).inverted() @ target[p].inverted() @ target[n]
        pb.matrix_basis = basis
        if keyframe is not None:
            pb.keyframe_insert("location", frame=keyframe)
            pb.keyframe_insert("rotation_quaternion", frame=keyframe)
    bpy.context.view_layer.update()


def evaluated_vertices(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    m = ev.to_mesh()
    co = np.zeros(len(m.vertices) * 3)
    m.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    return co.reshape(-1, 3) @ B


# --------------------------------------------------------------------------
def bake_pass(hero_obj, obj, tree, image_node, btype):
    """One selected-to-active bake hero -> game into `image_node`'s image (shared by the
    pattern variants, build_patterns.py, so every albedo is baked the same way)."""
    tree.nodes.active = image_node
    for o in bpy.context.selected_objects:
        o.select_set(False)
    hero_obj.select_set(True)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    kw = dict(type=btype, use_selected_to_active=True, cage_extrusion=.025, max_ray_distance=.06,
              margin=8, margin_type="EXTEND", use_clear=True)
    if btype == "DIFFUSE":
        kw["pass_filter"] = {"COLOR"}
    if btype == "NORMAL":
        kw["normal_space"] = "TANGENT"
    t0 = time.time()
    bpy.ops.object.bake(**kw)
    log("bake", btype, round(time.time() - t0, 1), "s")


def build_game(hero_obj, hero_rig, root):
    data = om.assemble("game")
    mat = bpy.data.materials.new("Octopus_game")
    obj = build_mesh(data, "Octopus_game_mesh", [mat])
    obj.data.uv_layers.remove(obj.data.uv_layers["Eye_UV"])
    uv = obj.data.uv_layers.new(name="UVMap")
    obj.data.uv_layers.active = uv
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=.004, area_weight=0.0)
    bpy.ops.uv.pack_islands(margin=.003, rotate=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    rig = build_rig(data, "Octopus_game_rig", obj)
    # --- bake hero -> game in the bind pose (both rigs at rest) -------------
    set_pose(hero_rig, om.BodyParams(), None, "hero")
    set_pose(rig, om.BodyParams(), None, "game")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    imgs = {}
    mat.use_nodes = True
    t = mat.node_tree
    for kind in ("albedo", "normal", "rough"):
        img = bpy.data.images.new(f"Octopus_game_{kind}", TEX, TEX, alpha=False, float_buffer=(kind == "normal"))
        if kind == "normal":
            img.colorspace_settings.name = "Non-Color"
        n = t.nodes.new("ShaderNodeTexImage")
        n.image = img
        imgs[kind] = (img, n)
    for kind, btype in (("albedo", "DIFFUSE"), ("normal", "NORMAL"), ("rough", "ROUGHNESS")):
        img, n = imgs[kind]
        bake_pass(hero_obj, obj, t, n, btype)
        log("baked", kind)
    alb = np.array(imgs["albedo"][0].pixels[:]).reshape(TEX, TEX, 4)
    rough = np.array(imgs["rough"][0].pixels[:]).reshape(TEX, TEX, 4)[..., 0]
    gloss = np.clip(1 - rough / .65, 0, 1)
    out_img = bpy.data.images.new("Octopus_albedo_gloss", TEX, TEX, alpha=True)
    alb[..., 3] = gloss
    out_img.pixels.foreach_set(alb.astype(np.float32).ravel())
    out_img.filepath_raw = str(OUT / "octopus_albedo_gloss.png")
    out_img.file_format = "PNG"
    out_img.save()
    nimg = imgs["normal"][0]
    nimg.filepath_raw = str(OUT / "octopus_normal.png")
    nimg.file_format = "PNG"
    nimg.save()
    # --- game material: opaque albedo in Blender, gloss spliced into the GLB --
    for nn in list(t.nodes):
        if nn.type == "TEX_IMAGE":
            t.nodes.remove(nn)
    bsdf = t.nodes["Principled BSDF"]
    opaque = bpy.data.images.new("Octopus_albedo_opaque", TEX, TEX, alpha=False)
    alb[..., 3] = 1
    opaque.pixels.foreach_set(alb.astype(np.float32).ravel())
    opaque.pack()
    tex = t.nodes.new("ShaderNodeTexImage")
    tex.image = opaque
    t.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt = t.nodes.new("ShaderNodeTexImage")
    nimg2 = bpy.data.images.load(str(OUT / "octopus_normal.png"))
    nimg2.colorspace_settings.name = "Non-Color"
    nt.image = nimg2
    nm = t.nodes.new("ShaderNodeNormalMap")
    t.links.new(nt.outputs["Color"], nm.inputs["Color"])
    t.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = .45
    # --- export GLB: two rest keys satisfy the loader's clip lookup ----------
    set_pose(rig, om.BodyParams(), None, "game", 1)
    set_pose(rig, om.BodyParams(), None, "game", 2)
    rig.animation_data.action.name = "Idle_Loop"
    scene.frame_start, scene.frame_end = 1, 2
    scene.frame_set(1)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    hero_obj.hide_set(True)
    bpy.ops.export_scene.gltf(filepath=str(OUT / "octopus_game.glb"), export_format="GLB",
                              use_selection=True, export_yup=True, export_animations=True,
                              export_animation_mode="ACTIVE_ACTIONS", export_force_sampling=True,
                              export_skins=True, export_def_bones=False, export_materials="EXPORT",
                              export_normals=True, export_tangents=False)
    hero_obj.hide_set(False)
    sys.path.insert(0, str(root / "art" / "characters"))
    from charkit.glbtex import replace_image
    replace_image(OUT / "octopus_game.glb", "baseColor", (OUT / "octopus_albedo_gloss.png").read_bytes())
    rig.animation_data_clear()
    np.savez_compressed(OUT / "octopus_game_geometry.npz", vertices=data["vertices"], joints=data["joints"],
                        weights=data["weights"], bind=data["bind"])
    return obj, rig, data


# --------------------------------------------------------------------------
def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def studio():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1400, 1000
    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (.16, .155, .15)
    scene.view_settings.view_transform = "Standard"
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, -.002))
    floor = bpy.context.object
    floor.name = "Studio_floor_not_exported"
    fm = bpy.data.materials.new("Studio_floor")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.47, .46, .44, 1)
    fm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = .8
    floor.data.materials.append(fm)
    for name, loc, power, size, color in [
        ("Key", (1.6, -2.2, 3.6), 230, 2.6, (1., .93, .85)),
        ("Fill", (-2.6, 1.0, 2.4), 90, 3.0, (.85, .9, 1.)),
        ("Top", (0., 0., 4.5), 120, 3.5, (1., 1., 1.)),
        ("Under", (0., 0., -3.5), 0, 4.0, (1., 1., 1.)),
    ]:
        d = bpy.data.lights.new(name, "AREA")
        d.energy, d.shape, d.size, d.color = power, "DISK", size, color
        o = bpy.data.objects.new(name, d)
        scene.collection.objects.link(o)
        o.location = loc
        aim(o, (0, 0, .1))
    cd = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam, floor


def shot(cam, path, loc, target, ortho=None, lens=50, res=None):
    scene = bpy.context.scene
    cam.location = Vector(loc)
    aim(cam, target)
    if ortho:
        cam.data.type, cam.data.ortho_scale = "ORTHO", ortho
    else:
        cam.data.type, cam.data.lens = "PERSP", lens
    if res:
        scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.filepath = str(path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    log("render", path.name, round(time.time() - t0, 1), "s")


def to_b(p):
    return tuple((B @ np.asarray(p, float)).tolist())


def renders(cam, floor, hero_obj, hero_rig, stage):
    scene = bpy.context.scene
    S = om.OCTOPUS_SCALE
    PREVIEW.mkdir(parents=True, exist_ok=True)
    poses = om.named_poses()
    body, arms = poses["rest"]
    set_pose(hero_rig, body, arms, "hero")
    c = (0, .15 * S, -.05 * S)
    if stage == "swim":
        set_pose(hero_rig, *poses["swim"], "hero")
        floor.hide_render = True
        shot(cam, PREVIEW / "pose_swim_3q.png", to_b((2.3 * S, 2.3 * S, 2.6 * S)), to_b((0, .15 * S, -.05 * S)), lens=50, res=(1200, 900))
        shot(cam, PREVIEW / "pose_swim_top.png", to_b((0, 6 * S, -.05 * S)), to_b(c), ortho=3.15 * S, res=(900, 900))
        floor.hide_render = False
        bpy.data.lights["Under"].energy = 140
        floor.hide_render = True
        shot(cam, PREVIEW / "underside.png", to_b((0, -6 * S, 0)), to_b((0, 0, 0)), ortho=3.15 * S, res=(1400, 1400))
        bpy.data.lights["Under"].energy = 0
        floor.hide_render = False
        stage = "gif"
    if stage in ("renders", "all"):
        shot(cam, PREVIEW / "top_down.png", to_b((0, 6 * S, -.05 * S)), to_b(c), ortho=3.15 * S, res=(1400, 1400))
        shot(cam, PREVIEW / "hero_three_quarter.png", to_b((2.3 * S, 2.1 * S, 2.6 * S)), to_b((0, .12 * S, -.05 * S)), lens=50, res=(1600, 1100))
        shot(cam, PREVIEW / "side_profile.png", to_b((4 * S, .25 * S, -.05 * S)), to_b((0, .25 * S, -.05 * S)), ortho=1.25 * S, res=(1400, 900))
        floor.hide_render = True
        bpy.data.lights["Under"].energy = 140
        shot(cam, PREVIEW / "underside.png", to_b((0, -6 * S, 0)), to_b((0, 0, 0)), ortho=3.15 * S, res=(1400, 1400))
        floor.hide_render = False
        shot(cam, PREVIEW / "junction_closeup.png", to_b((.75 * S, .55 * S, .75 * S)), to_b((.17 * S, .12 * S, .12 * S)), lens=60, res=(1400, 1000))
        floor.hide_render = True
        shot(cam, PREVIEW / "junction_underside.png", to_b((.55 * S, -.6 * S, .55 * S)), to_b((.1 * S, .07 * S, .1 * S)), lens=55, res=(1400, 1000))
        bpy.data.lights["Under"].energy = 0
        floor.hide_render = False
        for name, (bp, ap) in poses.items():
            set_pose(hero_rig, bp, ap, "hero")
            floor.hide_render = name == "swim"        # a swimming animal: no floor to clip the arms
            shot(cam, PREVIEW / f"pose_{name}_3q.png", to_b((2.3 * S, 2.3 * S, 2.6 * S)), to_b((0, .15 * S, -.05 * S)), lens=50, res=(1200, 900))
            shot(cam, PREVIEW / f"pose_{name}_top.png", to_b((0, 6 * S, -.05 * S)), to_b(c), ortho=3.15 * S, res=(900, 900))
    if stage in ("gif", "all"):
        floor.hide_render = False
        frames = PREVIEW / "swim_frames"
        frames.mkdir(exist_ok=True)
        floor.hide_render = True
        scene.cycles.samples = 20
        n = 30
        for i in range(n):
            bp, ap = om.swim_cycle(i * om.SWIM_PERIOD / n)
            set_pose(hero_rig, bp, ap, "hero")
            shot(cam, frames / f"{i:03d}.png", to_b((2.2 * S, 2.4 * S, 2.4 * S)), to_b((0, .12 * S, 0)), lens=45, res=(720, 540))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage", default="all", choices=["assets", "renders", "gif", "swim", "all"])
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    OUT.mkdir(parents=True, exist_ok=True)
    blend = OUT / "octopus_hero.blend"
    if args.stage in ("renders", "gif", "swim") and blend.exists():
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        gpu()
        hero_obj, hero_rig = bpy.data.objects["Octopus_hero_mesh"], bpy.data.objects["Octopus_hero_rig"]
        cam, floor = bpy.data.objects["Cam"], bpy.data.objects["Studio_floor_not_exported"]
        game = bpy.data.objects.get("Octopus_game_mesh")
        if game:
            game.hide_render = True
        hero_rig.animation_data_clear()
        renders(cam, floor, hero_obj, hero_rig, args.stage)
        log("OCTOPUS_RENDERS_OK")
        return
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    gpu()
    t0 = time.time()
    hero = om.assemble("hero")
    log("hero assembled", round(time.time() - t0, 1), "s")
    hero_obj = build_mesh(hero, "Octopus_hero_mesh", [skin_material(), cup_material(), eye_material()])
    hero_rig = build_rig(hero, "Octopus_hero_rig", hero_obj)
    hero_rig["prototype_note"] = ("Controller: scripts/octopus_math.py pose_octopus(body, arms). "
                                  "Keys on the timeline are samples of it, not the controller.")
    # parity: Blender's evaluated armature skin vs the shared reference, 3 poses
    parity = {}
    for name, (bp, apm) in list(om.named_poses().items())[:5]:
        set_pose(hero_rig, bp, apm, "hero")
        actual = evaluated_vertices(hero_obj)
        pts, rots = om.pose_octopus(bp, apm, "hero")
        expected = om.skin(hero["vertices"], hero["joints"], hero["weights"], hero["bind"], pts, rots)
        parity[name] = float(np.linalg.norm(actual - expected, axis=1).max())
    log("blender parity", parity)
    game_obj, game_rig, game = build_game(hero_obj, hero_rig, args.root.resolve())
    game_obj.hide_render = True
    game_obj.hide_set(True)
    game_rig.hide_set(True)
    # editable timeline on the hero: named poses held, then one swim cycle
    frame = 1
    marks = []
    for name, (bp, apm) in om.named_poses().items():
        set_pose(hero_rig, bp, apm, "hero", frame)
        set_pose(hero_rig, bp, apm, "hero", frame + 18)
        marks.append((name, frame))
        frame += 36
    swim0 = frame
    for i in range(25):
        bp, apm = om.swim_cycle(i * om.SWIM_PERIOD / 24)
        set_pose(hero_rig, bp, apm, "hero", frame + i * 3)
    hero_rig.animation_data.action.name = "Sampled_controls_inspection"
    for name, f in marks + [("swim", swim0)]:
        bpy.context.scene.timeline_markers.new(name, frame=f)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1, swim0 + 72
    for src in (HERE / "arm_math.py", HERE / "octopus_math.py", Path(__file__)):
        bpy.data.texts.load(str(src))
    cam, floor = studio()
    bpy.context.scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    count = lambda d: sum(len(f) - 2 for f in d["faces"])
    stats = dict(scale=om.OCTOPUS_SCALE, arm_length_m=arm.LENGTH * om.OCTOPUS_SCALE,
                 hero=dict(vertices=len(hero["vertices"]), triangles=count(hero), joints=len(hero["bones"]),
                           cups=int(8 * len(om.hero_arm()["cups"]))),
                 game=dict(vertices=len(game["vertices"]), triangles=count(game), joints=len(game["bones"]),
                           texture=f"{TEX}x{TEX} albedo+gloss(alpha) and tangent normal, baked from hero"),
                 blender_hero_skin_parity_m=parity)
    (OUT / "octopus_budget.json").write_text(json.dumps(stats, indent=2) + "\n")
    log("OCTOPUS_ASSETS_OK", json.dumps(stats))
    if args.stage == "all":
        hero_rig.animation_data_clear()
        renders(cam, floor, hero_obj, hero_rig, "all")


if __name__ == "__main__":
    main()
