"""Beatrice on a VRoid base (the VRoid spike, 2026-10-22): build and render from the committed CC0 VRM files.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/vroid.py -- [mode]
    (from art/characters)

    --candidates         step 1: every candidate VRM (vroid_tables.CANDIDATES, read from the download folder) imported,
                         its head rendered front / three-quarter as imported (Cycles, the studio rig) and in the toon look
    --out DIR            raw renders (default $TEMP/Beatrice_vroid)

Then `python Beatrice/scripts/vroid_sheets.py` lays out the pictures in `previews/`.
"""
import argparse
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import bpy  # noqa: E402
import numpy as np  # noqa: E402

import studio  # noqa: E402
import toonlook  # noqa: E402
import vrmbase  # noqa: E402
import vroid_tables as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", action="store_true")
ap.add_argument("--build", default="", help="B: build the variant and save it to source/Beatrice_vroid_<v>.blend")
ap.add_argument("--shots", default="", help="render shots of the saved build: a comma list of vroid_tables.SHOTS names")
ap.add_argument("--only", default="")
ap.add_argument("--open", default="", help="open a saved build (B) instead of building")
ap.add_argument("--face-test", action="store_true")
ap.add_argument("--pose", default="")
ap.add_argument("--whatif", default="", help="a WHAT-IF render: B with another VRoid model's hair (vroid_tables.WHATIF)")
ap.add_argument("--tag", default="B")
ap.add_argument("--out", default="")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
OUT = args.out or os.path.join(tempfile.gettempdir(), "Beatrice_vroid")
os.makedirs(OUT, exist_ok=True)


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vrmbase.enable()
    rig, cam, floor = studio.setup(samples=48)
    floor.hide_render = True
    return rig, cam


def head_frame(vrm):
    """The eyes' midpoint (world) from the VRM's eye bones (fallback: the head bone + 6 cm)."""
    try:
        return (vrm.bone("J_Adj_L_FaceEye") + vrm.bone("J_Adj_R_FaceEye")) / 2.0
    except Exception:
        return vrm.bone("C_Head") + np.array([0.0, 0.0, 0.06])


def head_shots(cam, eye, tag, spec, look=None):
    tx, ty, tz = eye[0], eye[1] - spec["back"], eye[2] + spec["up"]
    for az in spec["azimuths"]:
        a, e = math.radians(az), math.radians(spec["elev"])
        d = spec["dist"]
        cam.data.type, cam.data.lens = "PERSP", spec["lens"]
        cam.location = (tx + d * math.sin(a) * math.cos(e), ty - d * math.cos(a) * math.cos(e), tz + d * math.sin(e))
        studio._look_at(cam, (tx, ty, tz))
        if look:
            bpy.context.view_layer.update()
            toonlook.width_for(cam, (tx, ty, tz), spec["res"], look)
        studio.render(os.path.join(OUT, "%s_%03d.png" % (tag, az)), spec["res"], transparent=spec.get("transparent", False))


def candidates():
    only = [s for s in args.only.split(",") if s]
    for name, rel, cc0 in T.CANDIDATES:
        if only and name not in only:
            continue
        t0 = time.time()
        rig, cam = fresh()
        cam.parent = None
        vrm = vrmbase.load(os.path.join(T.DOWNLOADS, rel), "Cand")
        vrmbase.split_face(vrm)
        vrmbase.materials(vrm, T.PREFIX)
        eye = head_frame(vrm)
        print("candidate", name, "eyes at", eye.round(3), "height", vrmbase.rest_verts(vrm.parts["body"])[:, 2].max().round(3))
        head_shots(cam, eye, "cand_%s_neutral" % name, T.CANDIDATE_SHOTS)
        look = T.LOOK_VROID
        toonlook.apply(look, rig, cam, palette=T.PALETTE)
        head_shots(cam, eye, "cand_%s_toon" % name, T.CANDIDATE_SHOTS, look)
        print("candidate %s done in %.1fs" % (name, time.time() - t0))


def hair_b(vrm, spec):
    """The sample's hair made Beatrice's: the cat ears and the twin tails removed, the back and face-framing locks lengthened
    and draped, the lengthened hair weighted to the trunk by height."""
    ob = vrm.parts["hair"]
    me = ob.data
    V = vrmbase.rest_verts(ob)
    bones = vrm.arm.data.bones
    mat_of = np.zeros(len(me.vertices), int)
    for p in me.polygons:
        for v in p.vertices:
            mat_of[v] = p.material_index
    dead, back, side = [], [], []
    for idx in vrmbase.loose_parts(ob):
        P = V[idx]
        mname = me.materials[mat_of[idx[0]]].name
        g = vrmbase.dominant_group(ob, idx)
        b = bones.get(g) if g else None
        root = b if (b is not None and not b.name.startswith("J_Bip")) else None
        while root is not None and root.parent is not None and root.parent.name != "J_Bip_C_Head":
            root = root.parent
        rh = np.asarray(vrm.arm.matrix_world @ root.head_local) if root is not None else None
        if P[:, 2].max() > spec["ears"]["zmax"] or any(t in mname for t in spec["ears"]["materials"]):
            dead.append(idx)
        elif P[:, 2].min() < spec["tails"]["zmin"] and abs(P[:, 0].mean()) > spec["tails"]["xmin"]:
            dead.append(idx)
        elif P[:, 2].min() < 1.46 and P[:, 1].mean() > -0.005:
            back.append(idx)
        elif P[:, 2].min() < 1.46:
            side.append(idx)
    print("hair pieces: %d removed (ears, tails), %d back, %d side" % (len(dead), len(back), len(side)))
    # remove first, then find the pieces again (indices shift)
    keep_back = [V[i].mean(0) for i in back]
    keep_side = [V[i].mean(0) for i in side]
    vrmbase.delete_verts(ob, np.concatenate(dead))
    V = vrmbase.rest_verts(ob)
    parts = vrmbase.loose_parts(ob)
    cen = [V[i].mean(0) for i in parts]

    def match(cs):
        return [parts[int(np.argmin([np.linalg.norm(c - q) for q in cen]))] for c in cs]
    back, side = match(keep_back), match(keep_side)
    vrmbase.drape(ob, back, vrm.parts["body"], spec["back"], seed=1)
    vrmbase.drape(ob, side, vrm.parts["body"], spec["side"], seed=2)
    vrmbase.trunk_weights(ob, vrm, spec["trunk"])
    return ob


def face_b(vrm):
    """The face: its lower part longer (FACE_STRETCH), the expression set (EXPRESSION)."""
    face, lay = vrm.parts["face"], vrm.parts.get("face_layers")
    eye = (vrm.bone("J_Adj_L_FaceEye") + vrm.bone("J_Adj_R_FaceEye")) / 2.0
    F = vrmbase.rest_verts(face)
    chin = F[(np.abs(F[:, 0]) < 0.004) & (F[:, 1] < eye[1] - 0.02)][:, 2].min()
    k, fade = T.FACE_STRETCH["k"], T.FACE_STRETCH["fade"]
    lo = chin - fade

    def f(W):
        z = W[:, 2]
        d = np.clip(eye[2] - z, 0.0, eye[2] - chin) * k                 # the drop, growing below the eye line to the chin
        w = np.clip((z - lo) / fade, 0.0, 1.0)                           # fading out below the chin to the neck
        return z - d * (w * w * (3 - 2 * w))
    print("face: eye line %.4f, chin %.4f (eye-to-chin %.1f mm -> %.1f mm)" % (eye[2], chin, (eye[2] - chin) * 1000,
                                                                          (eye[2] - chin) * (1 + k) * 1000))
    for ob in (face, lay):
        if ob is not None:
            vrmbase.remap_z(ob, f)
    for ob in (face, lay):
        if ob is not None and ob.data.shape_keys:
            names = [kb.name for kb in ob.data.shape_keys.key_blocks]
            expression(ob, {s_: v for s_, v in T.EXPRESSION.items() if any(n.endswith(s_) for n in names)})


def pose_b(name):
    """Pose the opened build: `vroid_tables.POSE_B[name]`, the fingers relaxed, the standing foot on the floor."""
    vrm = opened()
    P = T.POSE_B[name]
    vrmbase.world_pose(vrm, P["rot"])
    curl_fingers(vrm)
    bpy.context.view_layer.update()
    V = vrmbase.evaluated_verts(vrm.parts["body"])
    side = -1.0 if P["stand"] == "R" else 1.0
    zmin = V[(V[:, 0] * side > 0.02) & (V[:, 2] < 0.3)][:, 2].min()
    vrm.arm.location.z -= zmin
    bpy.context.view_layer.update()
    return vrm


def curl_fingers(vrm):
    """Relax the fingers: each joint bent about its bone's local z (VRoid's finger bones point along their y)."""
    pb = vrm.arm.pose.bones
    for side in ("L", "R"):
        for fing, angles in T.FINGER_CURL.items():
            for i, a in enumerate(angles):
                b = pb.get("J_Bip_%s_%s%d" % (side, fing, i + 1))
                if b is None or not a:
                    continue
                b.rotation_mode = "XYZ"
                b.rotation_euler = (0.0, 0.0, math.radians(a) * (1.0 if side == "L" else -1.0))
    bpy.context.view_layer.update()


def opened():
    """The opened build as a vrmbase.Vrm."""
    arm = bpy.data.objects["Vroid_Armature"]
    parts = {n.split("_", 1)[1].lower(): o for n, o in bpy.data.objects.items() if n.startswith("Vroid_") and o.type == "MESH"}
    parts["face_layers"] = bpy.data.objects.get("Vroid_FaceLayers")
    return vrmbase.Vrm(arm, parts, "Vroid")


def landmark(vrm, name):
    """A posed landmark (world): `eyes` (the eye bones' midpoint), `crotch` (the body's lowest midline point), `chin` (the
    face's lowest front midline point)."""
    from mathutils import Vector
    arm = vrm.arm
    if name == "eyes":
        pb = arm.pose.bones
        return sum(((arm.matrix_world @ pb[n].head) for n in ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye")), Vector()) / 2.0
    if name == "nose":   # the face's front-most midline point
        ob = vrm.parts["face"]
        R = vrmbase.rest_verts(ob)
        i = int(np.argmin(np.where((np.abs(R[:, 0]) < 0.004) & (R[:, 2] > 1.40), R[:, 1], 9.0)))
        return Vector(vrmbase.evaluated_verts(ob)[i])
    if name == "crotch":
        ob = vrm.parts["body"]
        R = vrmbase.rest_verts(ob)
        i = int(np.argmin(np.where((np.abs(R[:, 0]) < 0.006) & (R[:, 2] > 0.5) & (R[:, 2] < 1.0), R[:, 2], 9.0)))
    else:
        ob = vrm.parts["face"]
        R = vrmbase.rest_verts(ob)
        e = landmark(vrm, "eyes")
        i = int(np.argmin(np.where((np.abs(R[:, 0]) < 0.004) & (R[:, 1] < -0.06) & (R[:, 2] > 1.38), R[:, 2], 9.0)))
    return Vector(vrmbase.evaluated_verts(ob)[i])


def framed_camera(vrm, cam, F):
    """Place the camera for a drawing's framing: its direction fixed (`F["dir"]`, from the target), its distance and aim
    solved so two posed landmarks (`F["marks"]`, default eyes and crotch) project to the drawing's pixels `F["px"]`."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    m1, m2 = (landmark(vrm, n) for n in F.get("marks", ("eyes", "crotch")))
    m3 = landmark(vrm, F["aim_mark"]) if "aim_mark" in F else None
    p1, p2 = (np.array(p, float) for p in F["px"])
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = F["res"]
    cam.data.type, cam.data.lens, cam.data.sensor_fit = "PERSP", F["lens"], "AUTO"
    u = Vector(F["dir"]).normalized()
    W, H = F["res"]

    def px(p):
        c = world_to_camera_view(sc, cam, p)
        return np.array([c.x * W, (1.0 - c.y) * H])

    tgt = (m1 + m2) / 2.0
    d = 2.5
    for _ in range(40):
        cam.location = tgt + u * d
        studio._look_at(cam, tuple(tgt))
        bpy.context.view_layer.update()
        e, c = px(m1), px(m2)
        if F.get("fit") == "vertical":   # a profile: the landmarks' depths differ from the drawing's, fit the height only
            d *= abs(c[1] - e[1]) / abs(p2[1] - p1[1])
        else:
            d *= np.linalg.norm(c - e) / np.linalg.norm(p2 - p1)
        mid_err = (e + c) / 2.0 - (p1 + p2) / 2.0
        if F.get("fit") == "vertical":   # and aim across by one landmark (`aim`: 0 the first, 1 the second), up and down
            # by the midpoint
            mid_err[0] = (e[0] - p1[0]) if F.get("aim", 0) == 0 else (c[0] - p2[0])
            if "aim_mark" in F:   # or by a third landmark (the nose's tip in a profile)
                mid_err[0] = px(m3)[0] - F["aim_px"][0]
        mpp = d * 36.0 / F["lens"] / max(W, H)
        right = Vector(cam.matrix_world.col[0][:3])
        up = Vector(cam.matrix_world.col[1][:3])
        tgt = tgt + right * (mid_err[0] * mpp) - up * (mid_err[1] * mpp)
    print("framed camera: distance %.3f m, marks at %s / %s px (wanted %s / %s)" % (d, px(m1).round(1), px(m2).round(1), p1, p2))
    return tuple(tgt)


def framed_shot(path, F):
    vrm = opened()
    cam = bpy.data.objects["Cam"]
    cam.parent = None
    tgt = framed_camera(vrm, cam, F)
    if LOOK:
        toonlook.width_for(cam, tgt, F["res"], LOOK)
    studio.render(path, F["res"], transparent=True)


def whatif_hair(src):
    """A WHAT-IF, not the build: B with another VRoid model's hair (`vroid_tables.WHATIF[src]`, read from the download
    folder, never committed -- its licence is not CC0). The hair is taken off its own skeleton, shifted to B's eyes, recoloured
    as B's, weighted to B's trunk by height and rigged to B's armature; B's own hair hidden."""
    W = T.WHATIF[src]
    vrm = opened()
    before = set(bpy.data.objects)
    other = vrmbase.load(os.path.join(T.DOWNLOADS, W["path"]), "Other")
    eye_o = (other.bone("J_Adj_L_FaceEye") + other.bone("J_Adj_R_FaceEye")) / 2.0
    hair = other.parts["hair"]
    keep = {hair}
    for o in [o for o in bpy.data.objects if o not in before and o not in keep]:
        bpy.data.objects.remove(o, do_unlink=True)
    # rest-space shift so its head meets B's (both are VRoid's default female head)
    arm = vrm.arm
    eye_b = sum((np.asarray(arm.matrix_world @ arm.data.bones[n].head_local) for n in ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye"))) / 2.0
    hair.parent = None
    V = vrmbase.rest_verts(hair) + (eye_b - eye_o)
    hair.matrix_world = vrm.parts["body"].matrix_world.copy()
    Mi = np.linalg.inv(np.asarray(hair.matrix_world))
    hair.data.vertices.foreach_set("co", (V @ Mi[:3, :3].T + Mi[:3, 3]).ravel())
    hair.data.update()
    for md in list(hair.modifiers):
        hair.modifiers.remove(md)
    hair.parent, hair.parent_type = vrm.parts["body"].parent, vrm.parts["body"].parent_type
    hair.matrix_parent_inverse = vrm.parts["body"].matrix_parent_inverse.copy()
    hair.matrix_world = vrm.parts["body"].matrix_world.copy()
    a = hair.modifiers.new("Armature", "ARMATURE")
    a.object = arm
    vrmbase.trunk_weights(hair, vrm, T.HAIR_B["trunk"])
    tmp = vrmbase.Vrm(arm, {"hair": hair}, "Other")
    vrmbase.materials(tmp, T.PREFIX, T.recolour())
    for o in bpy.data.objects:
        if o.name.startswith("Vroid_Hair"):
            o.hide_render = True
    hair.name = "Vroid_HairWhatif"
    print("what-if hair from %s: %d vertices, shifted %s m" % (src, len(hair.data.vertices), np.round(eye_b - eye_o, 4)))


def suit_b(vrm, S):
    """The leotard, stockings, zip and mesh inserts as two layers over the (subdivided) body, cut by `vroid_tables.suit_fields`."""
    body = vrm.parts["body"]
    vrmbase.subdivide(body, 2)
    f, crotch = T.suit_fields(vrm, vrmbase.rest_verts(body), S)
    pal = T.PALETTE
    latex = vrmbase.cut_material("%s_suit" % T.PREFIX, pal["SUIT"], ("suit", ("insert", -1.0)),
                                 lines=(("suit", S["piping"], S["piping_colour"], S["piping"]),
                                        ("insert", S["piping"], S["piping_colour"], -S["piping"]),
                                        ("zip", S["zip"][0], pal[S["zip_colour"]], 0.0)))
    net = vrmbase.cut_material("%s_mesh" % T.PREFIX, pal["V_SKIN"], ("insert",), net=S["net"])
    suit = vrmbase.shell(vrm, "Vroid_Suit", latex, S["offset"], {k: (lambda V, N, k=k: f(V, N)[k]) for k in ("suit", "insert", "zip")})
    vrmbase.rest_attr(suit)
    ins = vrmbase.shell(vrm, "Vroid_Insert", net, S["offset"] * 0.6, {"insert": lambda V, N: f(V, N)["insert"]})
    vrmbase.rest_attr(ins)
    vrm.parts["suit"], vrm.parts["insert"] = suit, ins
    C = S["collar_geo"]
    neck_y = vrm.bone("C_Neck")[1]

    def zipf(V, N):
        return np.where(V[:, 1] < neck_y, S["zip"][0] - np.abs(V[:, 0]), -1.0)
    vrm.parts["collar"] = vrmbase.collar(vrm, "Vroid_Collar", latex, C["z"][0], C["z"][1], C["gap"], C["flare"], C["front_drop"],
                                         thick=C["thick"], fields={"suit": lambda V, N: np.ones(len(V)),
                                                                   "insert": lambda V, N: -np.ones(len(V)), "zip": zipf})
    print("suit: crotch at %.3f m, body %d vertices" % (crotch, len(body.data.vertices)))


def build_b(save):
    t0 = time.time()
    rig, cam = fresh()
    vrm = vrmbase.load(T.BASE_B, "Vroid", "VRoid")
    vrmbase.split_face(vrm)
    vrmbase.drop(vrm)
    hair_b(vrm, T.HAIR_B)
    face_b(vrm)
    vrmbase.drop(vrm, T.DROP_FACE, roles=("face_layers",))
    vrmbase.materials(vrm, T.PREFIX, T.recolour())
    suit_b(vrm, T.SUIT_B)
    vrmbase.pose_bones(vrm, scale=T.SCALES)
    print("built B in %.1fs" % (time.time() - t0))
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=save)
    return vrm


def shots(names):
    """Render named shots of the opened build (vroid_tables.SHOTS)."""
    rig, cam = bpy.data.objects["StudioRig"], bpy.data.objects["Cam"]
    cam.parent = None
    for n in names:
        if n in T.FRAMES:
            framed_shot(os.path.join(OUT, "%s_%s.png" % (args.tag, n)), T.FRAMES[n])
            continue
        if n == "face":
            vrm = opened()
            pb = vrm.arm.pose.bones
            eye = sum((np.asarray(vrm.arm.matrix_world @ pb[k].head) for k in ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye"))) / 2.0
            head_shots(cam, eye, "%s_face" % args.tag, T.FACE_SHOT, LOOK)
            continue
        s = T.SHOTS[n]
        cam.data.type, cam.data.lens = "PERSP", s["lens"]
        cam.location = s["cam"]
        studio._look_at(cam, s["target"])
        if LOOK:
            bpy.context.view_layer.update()
            toonlook.width_for(cam, s["target"], s["res"], LOOK)
        studio.render(os.path.join(OUT, "%s_%s.png" % (args.tag, n)), s["res"], transparent=True)


def expression(vrm_face, keys):
    """Set the face's VRoid expression shape keys by their name's end (`Fcl_EYE_Angry` ...); every other key to 0."""
    kb = vrm_face.data.shape_keys.key_blocks
    for k in kb:
        if k.name != "Basis":
            k.value = 0.0
    for suffix, v in keys.items():
        hit = [k for k in kb if k.name.endswith(suffix)]
        if not hit:
            raise KeyError(suffix)
        hit[0].value = v


def face_test():
    face = bpy.data.objects["Vroid_Face"]
    lay = bpy.data.objects.get("Vroid_FaceLayers")
    for name, keys in T.FACE_TESTS.items():
        expression(face, keys)
        if lay is not None and lay.data.shape_keys:
            expression(lay, {k: v for k, v in keys.items() if any(kk.name.endswith(k) for kk in lay.data.shape_keys.key_blocks)})
        args.tag = "face_" + name
        shots(["t_face"])


LOOK = None
if args.candidates:
    candidates()
if args.open:
    bpy.ops.wm.open_mainfile(filepath=os.path.join(T.ROOT, "source", "Beatrice_vroid_%s.blend" % args.open))
    vrmbase.enable()
    if args.whatif:
        whatif_hair(args.whatif)
    _arm = bpy.data.objects["Vroid_Armature"]
    _c = np.asarray(_arm.matrix_world @ _arm.data.bones["J_Bip_C_Head"].head_local) + np.asarray(T.HEAD_CENTRE)
    toonlook.prepare(T.LOOK_VROID, ring_centre=tuple(_c))   # before posing: the hair's band, side attribute and normals
    if args.pose:
        pose_b(args.pose)
    LOOK = T.LOOK_VROID
    toonlook.apply(LOOK, bpy.data.objects["StudioRig"], bpy.data.objects["Cam"], palette=T.PALETTE)
    if args.face_test:
        face_test()
    elif args.shots:
        shots(args.shots.split(","))
if args.build:
    vrm = build_b(os.path.join(T.ROOT, "source", "Beatrice_vroid_%s.blend" % args.build))
    if args.shots:
        LOOK = T.LOOK_VROID
        toonlook.apply(LOOK, bpy.data.objects["StudioRig"], bpy.data.objects["Cam"], palette=T.PALETTE)
        shots(args.shots.split(","))
