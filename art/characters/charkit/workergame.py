"""The game-ready step shared by every worker character (Blender 4.5): `gameready.py` + `rig.py`
driven from a worker's own tables, plus the worker's evidence renders.

A worker's `scripts/game.py` puts `charkit` on the path and calls `run(...)`; its
`scripts/rig_spec.py` is `spec(worker)` unpacked into module names:

    blender -b --factory-startup -P <worker>/scripts/game.py -- [options]

    every `gameready.py` option (--tris, --voxel, --rig-only, --no-rig, ...), and
    --variant NAME   re-bake that look variant (worker.VARIANTS) onto the SAME skin and UVs:
                     game/albedo_NAME.png, copied beside the asset as <name>_NAME_albedo.png,
                     and a pose sheet of the rigged model wearing it (when the saved
                     game/<name>_rigged.blend exists)
    --retexture      re-bake the asset's own albedo (gloss in alpha) into its .glb, nothing else
                     (`gameready.retexture`)
    --evidence-only  skip the build and the rig: re-render the pose evidence from the saved
                     game/<name>_rigged.blend

Outputs: the asset `assets/models/<name>/<name>.glb`; `game/` as for every character; every
picture in `previews/game/` with an `index.html`; the `evidence` block of `game/stats.json`
(Walk_Loop floor penetration over every frame, per-pose lowest point and height).

Everything about the rig is DERIVED from the worker's tables (`spec`): joints, toe-out, the
arm abduction, the rigid head's cut line. Nothing here is per-character.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

# The clips' own upper-arm angle from vertical in Idle, as the marine's numbers fix it: modelled
# at 27 deg, given 12 deg of extra abduction (`space_marine_claude/scripts/rig_spec.py`).
CLIP_ARM_DEG = 15.0
# The coverall is 2.5 mm twill, the pockets 3-4 mm: walls under 1.5 voxels are thickened inward for
# the skin (the marine's open sheets get the same 1.5 voxels).
MIN_WALL_VOXELS = 1.5
# The coverall's trunk loft is thickened inward much further for the skin: the trouser hem stands up
# to 3.8 cm off the boot shaft inside it, an opening the cavity fill (2 voxels) does not close, so the
# legs stayed hollow and the decimated skin kept an inner surface and slits into it.
BODY_WALLS = (("Coverall_Body", 0.024),)
# Bare fingers stand 3-4 mm apart, under a voxel: at 6 mm they fuse into a mitten. The hands are
# remeshed apart at FINE_VOXEL into their own shells with their own triangle budget (out of the 10,000);
# the palm also enters the main skin, sunk PALM_INSET inside the fine one, to close the sleeve's cuff.
FINE_HANDS = dict(prefixes=("Hand_",), keep=("Hand_Palm",), voxel=0.0025, tris=1400)
PALM_INSET = (("Hand_Palm", 0.004),)
# A boot's toe bone takes over from the foot bone across this distance (m) at the toe cap's seam.
TOE_FADE_M = 0.02
# Head rigidity fades out over this distance (m) below the chin-to-jaw-angle line.
HEAD_FADE_M = 0.02

# The evidence poses: (clip, frame; -1 = last frame). Walk is shown at its widest stride.
POSES = (("Idle_Loop", 0), ("Walk_Loop", "stride"), ("Pistol_Shoot", 5), ("Death01", -1), ("Crouch_Idle_Loop", 0))
# Further poses for the deformation check: arms overhead, a deep kneel, a swing, sitting.
EXTREMES = (("Sword_Attack", 14), ("Fixing_Kneeling", 40), ("Punch_Cross", 10), ("Sitting_Idle_Loop", 0),
            ("Crouch_Fwd_Loop", 10), ("Jump_Loop", 6))


# ------------------------------------------------------------------- the rig spec
def _rows(table, key):
    return {r[0]: r for r in table[key]}


def _head_cy(head, z):
    """Centre y of the head loft at height z (HEAD rings)."""
    r = head["rings"]
    return float(np.interp(z, [x[0] for x in r], [x[1] for x in r]))


def _boot(dims):
    """(world point of a boot-frame point, sole top at y) -- `workwear.Boot`'s frame, no Blender."""
    s = dims["boot"]
    c, sn = math.cos(math.radians(s["toe_out"])), math.sin(math.radians(s["toe_out"]))
    BX, BY = np.array([c, sn, 0.0]), np.array([-sn, c, 0.0])
    ankle = np.array([s["ankle"][0], s["ankle"][1], 0.0])
    y_toe, y_heel = s["profile"][0][0], s["profile"][-1][0]
    w = lambda x, y, z: ankle + x * BX + y * BY + z * np.array([0.0, 0.0, 1.0])
    st = lambda y: s["sole"][0] + (s["sole"][1] - s["sole"][0]) * (y - y_toe) / (y_heel - y_toe)
    return w, st


def head_cut(head):
    """z of the line from the chin to the jaw angle (HEAD["jaw"]), as a function of y: above it the
    face, below it the throat and the neck."""
    cy, cz, ay, az = head["jaw"][:4]
    return lambda y: cz + (y - cy) * (az - cz) / (ay - cy)


def joints(dims, head, hair):
    """The rig's joints from the worker's tables (metres, faces -Y, +X its left; left side only)."""
    T, S = _rows(dims, "trunk"), _rows(dims, "sleeve")
    hip_z = T["hip"][1]

    def thigh_line(z):  # the leg loft's centre line through the knee and thigh rows, at height z
        a, b = np.array(T["knee"][1:4]), np.array(T["thigh"][1:4])  # (z, x, y)
        f = (z - a[0]) / (b[0] - a[0])
        return a[1:] + (b[1:] - a[1:]) * f

    cut = head_cut(head)
    y = _head_cy(head, cut(0.0))
    for _ in range(4):  # the head pivot: on the cut line, on the neck's centre
        y = _head_cy(head, cut(y))
    neck_z = T["yoke"][1]
    h = dims["hand"]
    L = np.array(h["down"], float)
    L /= np.linalg.norm(L)
    b = np.array(h["back"], float)
    Bv = b - (b @ L) * L
    Bv /= np.linalg.norm(Bv)
    W = np.cross(L, Bv)  # towards the thumb (`parts.hand`)
    wrist = np.array(S["cuff_end"][1:4]) + L * h["drop"]  # `workwear.build_hands`
    knuckle = wrist + h["scale"] * (0.094 * L + 0.011 * W - 0.002 * Bv)  # the middle finger's base (`parts.DIGITS`)
    w, st = _boot(dims)
    bt = dims["boot"]
    prof = bt["profile"]
    up0 = float(np.interp(0.0, [p[0] for p in prof], [p[2] for p in prof]))  # the upper's height over the ankle point
    y_toe = prof[0][0]
    tup = lambda v: tuple(float(x) for x in v)
    return dict(
        pelvis=(0.0, float(T["hip"][3]), hip_z),
        neck_base=(0.0, _head_cy(head, neck_z), neck_z),
        head_pivot=(0.0, y, cut(y)),
        head_top=(0.0, float(head["rings"][-1][1]), float(hair.get("crown_z", head["crown"][0]))),
        shoulder=tup(S["shoulder"][1:4]),
        elbow=tup(S["elbow"][1:4]),
        wrist=tup(wrist),
        knuckle=tup(knuckle),
        hip=tup(np.append(thigh_line(hip_z), hip_z)),
        knee=tup(T["knee"][2:4] + (T["knee"][1],)),
        ankle=tup(w(0.0, bt["shaft_y"], st(0.0) + 0.008 + 0.5 * up0)),
        ball=tup(w(0.0, bt["toe_cap_y"], st(bt["toe_cap_y"]) + 0.010)),
        toe_tip=tup(w(0.0, y_toe, st(y_toe) + 0.010)),
    )


def arm_angle_deg(J):
    """The modelled upper arm's angle out from vertical, in the frontal plane."""
    s, e = J["shoulder"], J["elbow"]
    return math.degrees(math.atan2(e[0] - s[0], s[2] - e[2]))


def gear(head, dims):
    """Rigid parts: the face rides the head bone (fading into the neck's bone heat below the
    chin-to-jaw line), hair, bun, ears and eyes ride it whole; hands their hand; a boot its foot
    behind the toe cap's seam and its toe bone in front of it (blended over TOE_FADE_M), so the
    toe stays on the floor when a clip rolls the foot onto its ball; the collar and undershirt
    the upper chest. The coverall deforms by bone heat."""
    cut = head_cut(head)
    bt = dims["boot"]
    c, sn = math.cos(math.radians(bt["toe_out"])), math.sin(math.radians(bt["toe_out"]))
    ankle = np.array(bt["ankle"], float)

    def boot(co, side):
        p = np.array([abs(co[0]), co[1]]) - ankle  # the left boot's frame (the right one is its mirror)
        y = -sn * p[0] + c * p[1]  # along the foot (BY), toe negative
        a = float(min(1.0, max(0.0, (bt["toe_cap_y"] - y) / TOE_FADE_M + 0.5)))
        return {"DEF-foot.{s}": 1.0 - a, "DEF-toe.{s}": a}

    def face(co, side):
        a = (co[2] - cut(co[1])) / HEAD_FADE_M + 1.0
        return {"DEF-head": float(min(1.0, max(0.0, a)))}

    return (
        ("Worker_Head", face),
        ("Worker_Hair", "DEF-head"), ("Worker_Ear", "DEF-head"), ("Worker_Eye", "DEF-head"),
        ("Hand", "DEF-hand.{s}"),
        ("Coverall_Cuff", "DEF-forearm.{s}"),
        ("Coverall_Collar", "DEF-spine.003"), ("Undershirt", "DEF-spine.003"),
        ("Boot_Shaft", None), ("Boot_Collar", None), ("Boot_Heel_Counter", None),  # the ankle bends by heat
        ("Boot", boot),
    )


def spec(worker):
    """The rig spec (`rig.py`'s JOINTS, FOOT_YAW_DEG, GEAR, ABDUCTION_DEG, TEST_POSES) of a worker."""
    J = joints(worker.DIMS, worker.HEAD, worker.HAIR)
    arm = arm_angle_deg(J)
    return dict(JOINTS=J, FOOT_YAW_DEG=float(worker.DIMS["boot"]["toe_out"]), GEAR=gear(worker.HEAD, worker.DIMS),
                ABDUCTION_DEG=max(0.0, arm - CLIP_ARM_DEG), MODELLED_ARM_DEG=arm,
                FLOOR_CLAMP=True,
                TEST_POSES=(("Idle_Loop", 0), ("Walk_Loop", 0), ("Walk_Loop", 16), ("Pistol_Shoot", 5),
                            ("Death01", -1), ("Crouch_Idle_Loop", 0)))


# ---------------------------------------------------------------------------- run
def _own_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    own = {"--evidence-only": "--evidence-only" in argv}
    if own["--evidence-only"]:
        sys.argv.remove("--evidence-only")
    own["variant"] = argv[argv.index("--variant") + 1] if "--variant" in argv else ""
    own["retexture"] = "--retexture" in argv
    own["no_previews"] = "--no-previews" in argv
    own["no_rig"] = "--no-rig" in argv
    return own


def run(root, name, worker, rig_spec):
    import gameready
    import workwear
    import garment

    own = _own_args()
    prev = os.path.join(root, "previews", "game")
    asset = os.path.join(REPO, "assets", "models", name, name + ".glb")

    def make_materials(variant=None):
        return lambda: workwear.materials(*worker.palette(variant))

    def build(M):
        workwear.build(M, worker.DIMS, worker.HEAD, worker.HAIR)
        garment.orient_outward()

    if not own["--evidence-only"]:
        gameready.run(root, name, build, make_materials(), worker.HEIGHT, worker.BEAUTY, rig_spec=rig_spec,
                      rig_out=asset, previews_dir=prev, variants=make_materials,
                      skin_opts=dict(min_wall_voxels=MIN_WALL_VOXELS, inset=PALM_INSET, fine=FINE_HANDS,
                                     walls=BODY_WALLS))
    if own["variant"]:
        variant_previews(root, name, own["variant"], prev)
    elif own["retexture"]:
        return  # the asset's albedo alone (gameready.retexture); the evidence renders are of the rig
    elif not own["no_rig"] and not own["no_previews"]:
        evidence(root, name, prev)
        write_index(root, name, prev)


# ----------------------------------------------------------------------- evidence
def _open_rigged(root, name):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.join(root, "game", name + "_rigged.blend"))
    T = bpy.data.objects["Rig"]
    skin = next(o for o in bpy.data.objects if o.type == "MESH")
    for m in skin.data.materials:  # the Workbench shows the ACTIVE image node: the albedo
        tex = [n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image.colorspace_settings.name == "sRGB"]
        if tex:
            m.node_tree.nodes.active = tex[0]
    return T, skin


def _clip(name):
    import bpy
    act = bpy.data.actions[name]
    return act, act.slots[0]


def _skin_co(skin):
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    ev = skin.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    return co.reshape(-1, 3)


def _workbench(px):
    import bpy
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type, sh.show_shadows, sh.show_cavity = "STUDIO", "TEXTURE", True, True
    sh.show_specular_highlight = False  # the baked albedo as it is, not the studio light's gloss
    sc.display.shadow_shift = 0.1
    sc.render.resolution_x, sc.render.resolution_y = px
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.color = (0.36, 0.375, 0.41)
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.quality = 90


def _shoot(cam_loc, target, path, px, ortho=None):
    import bpy
    import rig
    sc = bpy.context.scene
    cam = rig._camera("_cam", cam_loc, target, ortho=ortho, lens=50)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = px
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)


def _pose_frame(clip, frame, T):
    """Resolve a POSES frame: an int (-1 = last) or "stride", the frame of widest ankle spread."""
    import bpy
    act, slot = _clip(clip)
    import rig
    rig._assign(T, act, slot)
    f0, f1 = int(act.frame_start), int(act.frame_end)
    if frame == "stride":
        best = (-1.0, f0)
        for f in range(f0, f1):  # a loop's last frame repeats its first
            bpy.context.scene.frame_set(f)
            a, b = T.pose.bones["DEF-foot.L"].head, T.pose.bones["DEF-foot.R"].head
            best = max(best, (abs(a.y - b.y), f))
        return best[1]
    return f1 if frame < 0 else frame


def shoot_poses(T, skin, poses, prev, tag, px=(640, 820)):
    """Each pose: a front three-quarter view and straight down (the game's view), one file each."""
    import bpy
    import rig
    sc = bpy.context.scene
    floor = rig._floor()
    out = []
    for clip, frame in poses:
        f = _pose_frame(clip, frame, T)
        sc.frame_set(f)
        co = _skin_co(skin)
        c = (co.min(0) + co.max(0)) / 2
        files = {}
        for view in ("q", "top"):
            p = os.path.join(prev, "%s%s_%d_%s.jpg" % (tag, clip, f, view))
            if view == "q":
                # the character faces -Y: the camera stands in front, to its left
                k = max(1.0, float(np.ptp(co, 0).max()) / 1.6)  # a lying body: step back
                _shoot((c[0] + 1.75 * k, c[1] - 3.0 * k, c[2] + 0.85 * k), (c[0], c[1], c[2]), p, px)
            else:
                _shoot((c[0], c[1], 6.0), (c[0], c[1], 0.0), p, (px[0], px[0]), ortho=max(1.4, 1.15 * float(np.ptp(co[:, :2], 0).max())))
            files[view] = os.path.basename(p)
        out.append(dict(clip=clip, frame=f, files=files, min_z_mm=round(float(co[:, 2].min()) * 1e3, 1),
                        height_m=round(float(co[:, 2].max() - co[:, 2].min()), 3)))
    bpy.data.objects.remove(floor)
    return out


FACE_POSES = (("Idle_Loop", 0), ("Death01", -1), ("Sword_Attack", 14), ("Crouch_Idle_Loop", 0))


def shoot_faces(T, skin, prev, px=(560, 560)):
    """Close-ups of the head (a rigid face must not smear) and of the left hand, per pose: the camera
    in front of the head bone's face direction."""
    import bpy
    from mathutils import Vector
    sc = bpy.context.scene
    out = []
    for clip, frame in FACE_POSES:
        f = _pose_frame(clip, frame, T)
        sc.frame_set(f)
        hb = T.pose.bones["DEF-head"]
        rest = T.data.bones["DEF-head"].matrix_local.to_3x3()
        R = hb.matrix.to_3x3() @ rest.inverted()  # the head's rotation from its rest pose
        c = hb.head + (hb.tail - hb.head) * 0.35
        fwd = (R @ Vector((0.0, -1.0, 0.0))).normalized()  # the face looks down -Y at rest
        side = fwd.cross(Vector((0.0, 0.0, 1.0)))
        if side.length < 1e-3:
            side = R @ Vector((1.0, 0.0, 0.0))
        side.normalize()
        p = os.path.join(prev, "face_%s_%d.jpg" % (clip, f))
        _shoot(c + fwd * 0.75 + side * 0.35 + Vector((0.0, 0.0, 0.15)), c, p, px)
        hand = T.pose.bones["DEF-hand.L"]
        hc = hand.tail
        q = os.path.join(prev, "hand_%s_%d.jpg" % (clip, f))
        _shoot(hc + Vector((0.40, -0.40, 0.22)), hc, q, px)
        out.append(dict(clip=clip, frame=f, face=os.path.basename(p), hand=os.path.basename(q)))
    return out


def walk_floor(T, skin):
    """Deepest point of the skin under the floor over every Walk_Loop frame (mm) and where."""
    import bpy
    import rig
    act, slot = _clip("Walk_Loop")
    rig._assign(T, act, slot)
    worst = (1e9, None, None)
    per = []
    for f in range(int(act.frame_start), int(act.frame_end) + 1):
        bpy.context.scene.frame_set(f)
        co = _skin_co(skin)
        i = int(np.argmin(co[:, 2]))
        per.append(float(co[i, 2]))
        if co[i, 2] < worst[0]:
            worst = (float(co[i, 2]), f, co[i].tolist())
    return dict(deepest_mm=round(worst[0] * 1e3, 1), frame=worst[1], point_m=[round(x, 3) for x in worst[2]],
                frames=len(per), frames_below_minus_10mm=int(sum(z < -0.010 for z in per)))


def evidence(root, name, prev):
    t0 = time.time()
    T, skin = _open_rigged(root, name)
    _workbench((640, 820))
    os.makedirs(prev, exist_ok=True)
    poses = shoot_poses(T, skin, POSES, prev, "pose_")
    extremes = shoot_poses(T, skin, EXTREMES, prev, "extreme_")
    faces = shoot_faces(T, skin, prev)
    walk = walk_floor(T, skin)
    path = os.path.join(root, "game", "stats.json")
    with open(path) as f:
        stats = json.load(f)
    stats["evidence"] = dict(walk_loop_floor=walk, poses=poses, extremes=extremes, close_ups=faces,
                             timing_s=round(time.time() - t0, 1))
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)
    print("evidence:", json.dumps(stats["evidence"]), flush=True)


def variant_previews(root, name, variant, prev):
    """The rigged model wearing a variant's albedo (the same mesh), and the albedo beside the asset."""
    import shutil
    import bpy
    src = os.path.join(root, "game", "albedo_%s.png" % variant)
    dst = os.path.join(REPO, "assets", "models", name, "%s_%s_albedo.png" % (name, variant))
    shutil.copyfile(src, dst)
    if not os.path.isfile(os.path.join(root, "game", name + "_rigged.blend")):
        print("variant albedo copied to", dst, "-- no saved rigged .blend, so no pose sheet")
        return
    T, skin = _open_rigged(root, name)
    for m in skin.data.materials:
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image.colorspace_settings.name == "sRGB":
                n.image = bpy.data.images.load(src)
                n.image.colorspace_settings.name = "sRGB"
                m.node_tree.nodes.active = n
    _workbench((640, 820))
    shoot_poses(T, skin, (("Idle_Loop", 0), ("Walk_Loop", "stride")), prev, variant + "_")
    print("variant previews written; albedo copied to", dst)


def write_index(root, name, prev):
    with open(os.path.join(root, "game", "stats.json")) as f:
        s = json.load(f)
    ev, rg, me = s.get("evidence", {}), s.get("rig", {}), s.get("mesh", {})

    def imgs(rows):
        h = []
        for p in rows:
            h.append('<figure><img src="%s"><img src="%s"><figcaption>%s frame %d -- lowest %.1f mm, height %.3f m'
                     '</figcaption></figure>' % (p["files"]["q"], p["files"]["top"], p["clip"], p["frame"],
                                                 p["min_z_mm"], p["height_m"]))
        return "\n".join(h)

    extra = sorted(f for f in os.listdir(prev) if f.endswith((".png", ".jpg")) and not f.startswith(("pose_", "extreme_", "face_", "hand_")))
    html = """<!doctype html><meta charset="utf-8"><title>%s game model</title>
<style>body{font:14px sans-serif;background:#2b2d31;color:#ddd;margin:16px}img{max-width:100%%;height:auto;margin:2px}
figure{display:inline-block;margin:6px;vertical-align:top;max-width:660px}figure img{width:320px}pre{background:#1e1f22;padding:8px;overflow:auto}</style>
<h1>%s -- game-ready, rigged (Blender Workbench, the baked albedo)</h1>
<pre>%s</pre>
<h2>Test poses (front three-quarter | straight down)</h2>%s
<h2>Extreme poses (deformation check)</h2>%s
<h2>Close-ups: head and left hand</h2>%s
<h2>Other pictures</h2>%s
""" % (name, name, json.dumps(dict(triangles=me.get("triangles"), vertices_after_uv_split=me.get("vertices_after_uv_split"),
                                   bones=rg.get("bones"), clips=rg.get("clips"),
                                   clip_error=rg.get("clip_exactness_max_head_or_tail_mm"),
                                   glb_bytes=rg.get("export", {}).get("bytes"), walk_loop_floor=ev.get("walk_loop_floor"),
                                   silhouette_iou=s.get("silhouette_iou")), indent=1),
           imgs(ev.get("poses", [])), imgs(ev.get("extremes", [])),
           "\n".join('<figure><img src="%s"><img src="%s"><figcaption>%s frame %d</figcaption></figure>'
                     % (c["face"], c["hand"], c["clip"], c["frame"]) for c in ev.get("close_ups", [])),
           "\n".join('<figure><img src="%s"><figcaption>%s</figcaption></figure>' % (f, f) for f in extra))
    with open(os.path.join(prev, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

