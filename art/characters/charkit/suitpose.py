"""Pose a built bodysuit character for stills (Blender 4.5, Cycles): B1m, Beatrice's own.

A render-time step on a COPY of the saved build (`source/<name>.blend`): the build itself is never
changed. The character's `scripts/posed.py` puts `charkit` on the path and calls `run(...)` with its
tables module, whose `POSE` table holds everything about her (joints, weight bands, poses, cameras):

    blender -b --factory-startup -P <name>/scripts/posed.py -- [options]

    --pose NAME      a pose in tables.POSE["poses"] (default: the first)
    --views a,b,..   shots in tables.POSE["shots"] (default: every shot not marked `dev`); `rest:<shot>` renders the
                     unposed (frozen) build in that shot
    --lights NAME    an extra light setup from tables.POSE["lights"] for every shot (default: the studio rig)
    --draft          quarter resolution, few samples (the working loop)
    --samples N      Cycles samples (default 96)
    --blend PATH     the build to pose (default source/<name>.blend)
    --save PATH      also save the posed scene
    --out DIR        where the raw RGBA renders go (default $TEMP/<name>_posed)

How it poses (a skeleton of our own; the game's 53-bone rig binds only the fused game skin):

1. freeze -- every part's modifier stack applied (`new_from_object`, every layer kept: custom normals, the
   piping and mesh attributes), its transform baked in, so each vertex is in world space;
2. skeleton -- bones with world-aligned rest frames at the table's joints (pelvis, two spine bones,
   neck, head; per side upper arm, forearm, hand, thigh, shin, foot). A pose gives bones their WORLD
   rotation (degrees about the world X / Y / Z, applied in that order); a bone it leaves out turns with
   its parent. Forward kinematics carries each bone's head with its parent;
3. weights -- ONE smooth field of position (C2 ramps, `smoother`), evaluated at every vertex of every part,
   so a cord, a seam, the zip or an insert moves exactly as the suit under it: the trunk shared between
   pelvis / spine / chest / neck / head by height; an arm owns what lies near its own chain (a distance
   to the arm's polyline, its fade WIDER where the arm still meets the trunk, `arm_radius_near`: a narrow
   fade there folded the flank under the armpit) beyond a ramp across the shoulder joint; a leg owns what lies below a ramp
   across the hip joint's plane, faded to nothing at the mid-plane (the crotch moves with the pelvis);
   along each limb, ramps across the elbow / wrist / knee / ankle planes. Arms win over legs, legs
   over the trunk. Rigid parts (`rigid`: knee pads, zip pulls) take the field's value at their centre
   for every vertex, so they turn whole;
4. skinning -- linear blend of the bones' rigid transforms, in numpy; then the figure is lowered so
   the standing foot's sole is back on the floor.
"""
import argparse
import fnmatch
import math
import os
import sys
import tempfile
import time

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

import studio


def smoother(x, a, b):
    """0 below a, 1 above b, C2 (6t^5 - 15t^4 + 10t^3) between."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


# ------------------------------------------------------------------- skeleton
TRUNK = ("pelvis", "spine", "chest", "neck", "head")
ARM = ("upper_arm", "forearm", "hand")
LEG = ("thigh", "shin", "foot")


class Skeleton:
    """Bones with world-aligned rest frames. `J` (the table's `joints`): trunk joints by bone name, side
    joints for her LEFT (x > 0) by landmark, mirrored for the right."""

    def __init__(self, J):
        self.J = {k: np.array(v, float) for k, v in J.items()}
        self.head, self.parent = {}, {}
        for i, b in enumerate(TRUNK):
            self.head[b] = self.J[b]
            self.parent[b] = TRUNK[i - 1] if i else None
        for s, k in (("L", 1.0), ("R", -1.0)):
            m = np.array([k, 1.0, 1.0])
            for b, j, p in (("upper_arm", "shoulder", "chest"), ("forearm", "elbow", "upper_arm"), ("hand", "wrist", "forearm"),
                            ("thigh", "hip", "pelvis"), ("shin", "knee", "thigh"), ("foot", "ankle", "shin")):
                self.head[b + "." + s] = self.J[j] * m
                self.parent[b + "." + s] = p if p in TRUNK else p + "." + s
        self.names = list(self.head)

    def side_points(self, s):
        """The left chains' landmark points mirrored to side s."""
        m = np.array([1.0 if s == "L" else -1.0, 1.0, 1.0])
        return {k: v * m for k, v in self.J.items()}

    def pose(self, rot, offset=(0.0, 0.0, 0.0)):
        """rot: bone -> world rotation (deg about X, Y, Z, applied in that order). Returns bone -> (R 3x3, rest head, posed head)."""
        out = {}
        for b in self.names:  # parents come first (TRUNK, then each side's chain in order)
            p = self.parent[b]
            if b in rot:
                R = np.array(Euler([math.radians(a) for a in rot[b]], "XYZ").to_matrix())
            else:
                R = out[p][0] if p else np.eye(3)
            if p is None:
                h = self.head[b] + np.array(offset, float)
            else:
                Rp, hp0, hp1 = out[p]
                h = Rp @ (self.head[b] - hp0) + hp1
            out[b] = (R, self.head[b], h)
        return out


# --------------------------------------------------------------------- weights
def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _unit(v):
    return v / np.linalg.norm(v)


def weights(P, sk, W):
    """P (n, 3) world points at rest -> {bone: (n,) weight}, summing to 1. `W`: the table's `weights` bands."""
    z = P[:, 2]
    s = [smoother(z, *W["trunk"][i]) for i in range(4)]
    trunk = {"pelvis": 1.0 - s[0], "spine": s[0] - s[1], "chest": s[1] - s[2], "neck": s[2] - s[3], "head": s[3]}
    J = sk.J
    out = {b: np.zeros(len(P)) for b in sk.names}
    arm_sum, leg_sum, limbs = np.zeros(len(P)), np.zeros(len(P)), []
    for side, k in (("L", 1.0), ("R", -1.0)):
        Q = P * np.array([k, 1.0, 1.0])
        # the arm: near its own chain, beyond the shoulder joint's ramp
        u_up, u_fore = _unit(J["elbow"] - J["shoulder"]), _unit(J["wrist"] - J["elbow"])
        chain = [J["shoulder"], J["elbow"], J["wrist"], J["knuckle"], J["fingertip"]]
        d = np.min([_seg_dist(Q, a, b) for a, b in zip(chain[:-1], chain[1:])], axis=0)
        t_up = (Q - J["shoulder"]) @ u_up
        r0, r1 = W["arm_radius"]
        if "arm_radius_near" in W:  # where the arm still meets the trunk (the armpit) the radial fade is wider: a smooth stretch, no fold
            r1n, ta, tb = W["arm_radius_near"]
            r1 = r1 + (r1n - r1) * (1.0 - smoother(t_up, ta, tb))
        fade = np.clip((d - r0) / (r1 - r0), 0.0, 1.0)
        fade = fade * fade * fade * (fade * (6.0 * fade - 15.0) + 10.0)
        arm = smoother(t_up, *W["arm_root"]) * (1.0 - fade) * smoother(Q[:, 0], *W["arm_side"])
        f = smoother((Q - J["elbow"]) @ u_up, *W["elbow"])
        h = smoother((Q - J["wrist"]) @ u_fore, *W["wrist"])
        # the leg: below the hip joint's plane, faded to nothing at the mid-plane
        u_th, u_sh = _unit(J["knee"] - J["hip"]), _unit(J["ankle"] - J["knee"])
        leg = smoother((Q - J["hip"]) @ u_th, *W["leg_root"]) * smoother(Q[:, 0], *W["leg_mid"])
        kn = smoother((Q - J["knee"]) @ u_th, *W["knee"])
        an = smoother((Q - J["ankle"]) @ u_sh, *W["ankle"])
        limbs.append((side, arm, (1.0 - f, f * (1.0 - h), f * h), leg, (1.0 - kn, kn * (1.0 - an), kn * an)))
        arm_sum += arm
        leg_sum += leg
    rest_arm = np.clip(1.0 - arm_sum, 0.0, 1.0)
    rest_leg = np.clip(1.0 - leg_sum, 0.0, 1.0)
    for side, arm, ach, leg, lch in limbs:
        for b, w in zip(ARM, ach):
            out[b + "." + side] += arm * w
        for b, w in zip(LEG, lch):
            out[b + "." + side] += rest_arm * leg * w
    for b, w in trunk.items():
        out[b] += rest_arm * rest_leg * w
    return out


# ----------------------------------------------------------------- the scene
def freeze():
    """Every mesh part with its modifiers applied and its transform baked in (world-space vertices)."""
    dg = bpy.context.evaluated_depsgraph_get()
    obs = [o for o in bpy.data.objects if o.type == "MESH" and o.name != "Floor"]
    for ob in obs:
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        old = ob.data
        ob.modifiers.clear()
        ob.data = me
        mw = ob.matrix_world.copy()
        ob.parent = None
        me.transform(mw)
        ob.matrix_world = Matrix.Identity(4)
        if old.users == 0:
            bpy.data.meshes.remove(old)
    return obs


def _co(ob):
    co = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def deform(obs, sk, posed, W, rigid=(), attach=None):
    """Linear-blend skinning of every part (world-space vertices, see `freeze`). `attach` = {pattern: bone}: parts that
    move wholly with one bone (B2a: the hair with the head)."""
    for ob in obs:
        P = _co(ob)
        bone = next((b for pat, b in (attach or {}).items() if fnmatch.fnmatch(ob.name, pat)), None)
        if bone is not None:
            w = {b: (np.ones(len(P)) if b == bone else np.zeros(len(P))) for b in sk.names}
        elif any(fnmatch.fnmatch(ob.name, pat) for pat in rigid):
            # each side whole (a mirrored part is one object holding both), at the field's value at that side's centre
            w = {b: np.zeros(len(P)) for b in sk.names}
            for m in (P[:, 0] >= 0.0, P[:, 0] < 0.0):
                if m.any():
                    for b, v in weights(P[m].mean(0, keepdims=True), sk, W).items():
                        w[b][m] = v[0]
        else:
            w = weights(P, sk, W)
        Q = np.zeros_like(P)
        for b, wb in w.items():
            if not np.any(wb > 1e-9):
                continue
            R, h0, h1 = posed[b]
            Q += wb[:, None] * ((P - h0) @ R.T + h1)
        ob.data.vertices.foreach_set("co", Q.ravel())
        ob.data.update()


def _side_z(obs, part):
    """The vertices' heights of the parts matching part = (pattern, side): side "L" keeps x > 0, "R" x < 0 (a mirrored part
    is one object holding both sides)."""
    pattern, side = part
    zs = []
    for o in obs:
        if fnmatch.fnmatch(o.name, pattern):
            P = _co(o)
            zs.append(P[P[:, 0] > 0.0, 2] if side == "L" else P[P[:, 0] < 0.0, 2])
    return np.concatenate(zs)


def drop_to_floor(obs, part, z_floor=0.0):
    """Lower (or raise) everything so the lowest vertex of `part` (pattern, side) sits at z_floor; returns the shift."""
    dz = z_floor - _side_z(obs, part).min()
    for ob in obs:
        P = _co(ob)
        P[:, 2] += dz
        ob.data.vertices.foreach_set("co", P.ravel())
        ob.data.update()
    return dz


def lowest(obs, part):
    return float(_side_z(obs, part).min())


def add_lights(spec, rig):
    """An extra light setup (tables.POSE["lights"][name]): the studio's own lights scaled by `studio_scale`, plus
    area lights (name, location, target, size (x, y), energy, colour), rectangles so a glossy surface throws a streak."""
    for ob in bpy.data.objects:
        if ob.type == "LIGHT" and ob.parent == rig:
            ob.data.energy *= spec.get("studio_scale", 1.0)
    for name, loc, target, size, energy, col in spec.get("areas", ()):
        li = bpy.data.lights.new(name, "AREA")
        li.shape = "RECTANGLE"
        li.size, li.size_y = size
        li.energy, li.color = energy, col
        ob = bpy.data.objects.new(name, li)
        ob.location = loc
        studio._look_at(ob, target)
        bpy.context.scene.collection.objects.link(ob)
    if "world_scale" in spec:
        nt = bpy.context.scene.world.node_tree
        for n in nt.nodes:
            if n.type == "MAP_RANGE":
                n.inputs["To Min"].default_value *= spec["world_scale"]
                n.inputs["To Max"].default_value *= spec["world_scale"]


def shoot(cam, rig, shot, path, scale=1.0, transparent=False):
    """A shot: dict(cam=(x, y, z), target=(x, y, z), lens=mm, res=(w, h)[, rig=deg]) -- the camera placed
    in the world; `rig` turns the studio rig (its lights) about the vertical first."""
    rig.rotation_euler = (0.0, 0.0, math.radians(shot.get("rig", 0.0)))
    bpy.context.view_layer.update()
    cam.parent = None
    cam.data.type = "PERSP"
    cam.data.lens = shot["lens"]
    cam.data.sensor_fit = "AUTO"
    cam.data.clip_start, cam.data.clip_end = 0.05, 60.0
    cam.location = shot["cam"]
    studio._look_at(cam, shot["target"])
    res = shot["res"]
    studio.render(path, (int(res[0] * scale), int(res[1] * scale)), transparent=transparent)
    rig.rotation_euler = (0.0, 0.0, 0.0)


def use_gpu():
    """The GPU, as studio.setup picks it (a .blend opened under --factory-startup keeps the scene's device but not the
    preferences' device type, so it would render on the CPU)."""
    sc = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            (getattr(prefs, "refresh_devices", None) or prefs.get_devices)()
            devs = [d for d in prefs.devices if d.type == kind]
            for d in prefs.devices:
                d.use = d.type == kind
            if devs:
                sc.cycles.device = "GPU"
                return kind
        except Exception:
            continue
    sc.cycles.device = "CPU"
    return "CPU"


def run(root, name, tables):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="")
    ap.add_argument("--views", default="")
    ap.add_argument("--lights", default="")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--blend", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    T = tables.POSE
    t0 = time.time()
    bpy.ops.wm.open_mainfile(filepath=args.blend or os.path.join(root, "source", name + ".blend"))
    sc = bpy.context.scene
    sc.cycles.samples = args.samples or (16 if args.draft else 96)
    print("render device:", use_gpu())
    rig, cam = bpy.data.objects["StudioRig"], bpy.data.objects["Cam"]
    obs = freeze()
    t1 = time.time()
    print("frozen in %.1fs" % (t1 - t0))
    pose_name = args.pose or next(iter(T["poses"]))
    views = [v for v in args.views.split(",") if v] or [k for k, v in T["shots"].items() if not v.get("dev")]
    rest = all(v.startswith("rest") for v in views)
    if not rest:
        P = T["poses"][pose_name]
        sk = Skeleton(T["joints"])
        posed = sk.pose(P["rot"], P.get("offset", (0.0, 0.0, 0.0)))
        deform(obs, sk, posed, T["weights"], T.get("rigid", ()), T.get("attach"))
        dz = drop_to_floor(obs, P["stand"])
        print("posed '%s' in %.1fs (lowered %.1f mm; the free foot's sole %.1f mm above the floor)" % (
            pose_name, time.time() - t1, -dz * 1000, lowest(obs, P["free"]) * 1000))
    if args.lights:
        add_lights(T["lights"][args.lights], rig)
    out = args.out or os.path.join(tempfile.gettempdir(), name + "_posed")  # raw RGBA renders; the sheets are laid out from here
    os.makedirs(out, exist_ok=True)
    tag = pose_name + ("_" + args.lights if args.lights else "")
    for v in views:
        shot = T["shots"][v.split(":")[1] if v.startswith("rest:") else ("concept" if v == "rest" else v)]
        t2 = time.time()
        path = os.path.join(out, "%s_%s.png" % ("rest" if rest else tag, v.replace("rest:", "")))
        shoot(cam, rig, shot, path, 0.25 if args.draft else 1.0, transparent=True)
        print("shot %s in %.1fs -> %s" % (v, time.time() - t2, path))
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=args.save)
    print("done in %.1fs" % (time.time() - t0))
