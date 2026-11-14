"""Rig stage of `gameready.py`: the game skin bound to the game's skeleton, with the game's clips.

The game has ONE humanoid skeleton and one clip library (`assets/models/marine/`, Quaternius
Universal Animation Library, 53 bones, 46 clips; `docs/reference/adding_character_models.md`).
Its rest pose is a T-pose; our characters are modelled arms-down. The skin is never bent into the
T-pose (linear-blend un-posing pinches the shoulders). Instead the character gets its own rest pose
and the clips are converted to it:

1. landmarks -- the character's joints in its own pose (its `scripts/rig_spec.py`, from its model's
   numbers); the spine is spread between pelvis and neck base in the source's proportions;
2. driver rig D -- the source rig re-proportioned: every bone keeps its rest direction and roll,
   only lengths and offsets change. D still rests in the T-pose, so the source clips play on it
   unchanged, except `root` and `DEF-hips` location, scaled by the leg-length ratio;
3. fit pose -- D posed by pure rotations (each bone the shortest arc from its parent-carried
   direction to its landmark) so its joints land on the landmarks;
4. target rig T -- D's fit-pose matrices made the rest pose: same 53 names and hierarchy;
5. clips -- each source clip played on D and baked onto T so every T bone's armature-space matrix
   equals D's, frame for frame (direct matrix math on Blender's evaluated D);
6. weights -- bone heat on the untouched skin at T's rest, then rigid overrides for hard gear:
   every skin vertex is labelled with its nearest high-res source part and the spec's GEAR table
   maps parts to bones; at most 4 influences, normalised;
7. export -- a skinned .glb, Y-up, identity object transforms, the clips under their own names.

Starts from the saved P1 file `<character>/game/<name>_game.blend` (`gameready.py --rig-only`).
"""
import json
import math
import os
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

import gameready
import kit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SOURCE_RIG = os.path.join(REPO, "assets", "models", "marine", "AnimationLibrary_Godot_Standard.gltf")

SPINE = ("DEF-spine.001", "DEF-spine.002", "DEF-spine.003")
# Joint (landmark) -> the bone whose HEAD sits there. Side joints carry ".{s}".
HEAD_AT = {"pelvis": "DEF-hips", "neck_base": "DEF-neck", "head_pivot": "DEF-head",
           "shoulder": "DEF-upper_arm.{s}", "elbow": "DEF-forearm.{s}", "wrist": "DEF-hand.{s}",
           "knuckle": "DEF-f_middle.01.{s}", "hip": "DEF-thigh.{s}", "knee": "DEF-shin.{s}",
           "ankle": "DEF-foot.{s}", "ball": "DEF-toe.{s}"}
# Bone -> what it points at in the fit pose: a bone (its head) or a joint the bone's TAIL reaches.
AIM = {"DEF-hips": "DEF-spine.001", "DEF-spine.001": "DEF-spine.002", "DEF-spine.002": "DEF-spine.003",
       "DEF-spine.003": "DEF-neck", "DEF-neck": "DEF-head", "DEF-head": ":head_top",
       "DEF-shoulder.{s}": "DEF-upper_arm.{s}", "DEF-upper_arm.{s}": "DEF-forearm.{s}",
       "DEF-forearm.{s}": "DEF-hand.{s}", "DEF-hand.{s}": "DEF-f_middle.01.{s}",
       "DEF-thigh.{s}": "DEF-shin.{s}", "DEF-shin.{s}": "DEF-foot.{s}", "DEF-foot.{s}": "DEF-toe.{s}",
       "DEF-toe.{s}": ":toe_tip"}
# The feet are fitted FLAT: their fit rotation is the character's toe-out yaw alone, so a boot modelled
# flat on the floor stays flat wherever a clip puts the source's (flat) foot. Their landmarks set lengths.
FLAT = ("DEF-foot.{s}", "DEF-toe.{s}")
MOVING = ("root", "DEF-hips")  # the only bones whose location the clips animate
TEST_POSES = (("Idle_Loop", 0), ("Walk_Loop", 0), ("Walk_Loop", 16), ("Pistol_Shoot", 5), ("Death01", -1))


def _sides(table):
    out = {}
    for k, v in table.items():
        if "{s}" in k or "{s}" in v:
            for s in ("L", "R"):
                out[k.format(s=s)] = v.format(s=s)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------- source rig
def import_source():
    objs, acts = set(bpy.data.objects), set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=SOURCE_RIG)
    new = [o for o in bpy.data.objects if o not in objs]
    src = next(o for o in new if o.type == "ARMATURE")
    for o in new:
        if o is not src:
            bpy.data.objects.remove(o)
    actions = sorted((a for a in bpy.data.actions if a not in acts), key=lambda a: a.name)
    assert src.matrix_world == Matrix.Identity(4), "source rig is not at identity"
    return src, actions


def _channelbag(action, ensure_slot=None):
    for layer in action.layers:
        for strip in layer.strips:
            for slot in action.slots:
                cb = strip.channelbag(slot)
                if cb is not None:
                    return cb, slot
    return None, None


def _new_action(name, id_name):
    act = bpy.data.actions.new(name)
    slot = act.slots.new(id_type="OBJECT", name=id_name)
    strip = act.layers.new("Layer").strips.new(type="KEYFRAME")
    return act, slot, strip.channelbag(slot, ensure=True)


def _set_curve(cb, path, index, frames, values, group=None):
    fc = cb.fcurves.find(path, index=index)
    if fc is None:
        fc = cb.fcurves.new(path, index=index)
        if group is not None:
            g = cb.groups.get(group) or cb.groups.new(group)
            fc.group = g
    kp = fc.keyframe_points
    kp.clear()
    kp.add(len(frames))
    kp.foreach_set("co", np.column_stack([frames, values]).astype(np.float32).ravel())
    kp.foreach_set("interpolation", np.ones(len(frames), np.int32))  # LINEAR
    fc.update()


def _assign(ob, act, slot):
    ad = ob.animation_data or ob.animation_data_create()
    ad.action = act
    ad.action_slot = slot


def _clear_anim(ob):
    if ob.animation_data:
        for t in list(ob.animation_data.nla_tracks):
            ob.animation_data.nla_tracks.remove(t)
        ob.animation_data.action = None
    for pb in ob.pose.bones:
        pb.location, pb.rotation_quaternion, pb.scale = (0, 0, 0), (1, 0, 0, 0), (1, 1, 1)


def _copy_rig(src, name):
    ob = src.copy()
    ob.data = src.data.copy()
    ob.name = ob.data.name = name
    bpy.context.scene.collection.objects.link(ob)
    _clear_anim(ob)
    return ob


def _set_rest(rig, mats, lengths):
    gameready._select([rig], rig)
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in rig.data.edit_bones:
        eb.use_connect = False
    for name, M in mats.items():
        eb = rig.data.edit_bones[name]
        eb.matrix = M
        eb.length = lengths[name]
    bpy.ops.object.mode_set(mode="OBJECT")


# ------------------------------------------------------------------------- plan
def landmarks(spec):
    """Joint name -> world position (both sides), spine joints included."""
    J = {}
    for k, v in spec.JOINTS.items():
        p = Vector(v)
        if abs(p.x) < 1e-9:
            J[k] = p
        else:
            J[k + ".L"] = p
            J[k + ".R"] = Vector((-p.x, p.y, p.z))
    return J


def plan(src, spec):
    """D's rest (head, length per bone), the fit pose's world rotations Q and heads, from the landmarks."""
    J = landmarks(spec)
    bones = src.data.bones
    head_at = {}
    for j, b in HEAD_AT.items():
        if "{s}" in b:
            for s in ("L", "R"):
                head_at[b.format(s=s)] = J[j + "." + s]
        else:
            head_at[b] = J[j]
    # spine joints: spread between pelvis and neck base in the source's proportions
    s0, s1 = bones["DEF-hips"].head_local, bones["DEF-neck"].head_local
    for b in SPINE:
        f = (bones[b].head_local.z - s0.z) / (s1.z - s0.z)
        head_at[b] = J["pelvis"].lerp(J["neck_base"], f)
    aim = _sides(AIM)
    tails = {b: J[a[1:]] if a[1:] in J else J[a[1:] + "." + b[-1]] for b, a in aim.items() if a.startswith(":")}

    flat = {b.format(s=s) for b in FLAT for s in ("L", "R")}
    yaw = math.radians(getattr(spec, "FOOT_YAW_DEG", 0.0))
    order = sorted(bones, key=lambda b: len(b.parent_recursive))
    Q, Hf, Hd, sc, Ld, fit_err = {}, {}, {}, {}, {}, {}
    for b in order:
        n, p = b.name, b.parent
        if p is None:
            Q[n], Hf[n], Hd[n], sc[n] = Matrix.Identity(3), b.head_local.copy(), b.head_local.copy(), 1.0
        else:
            pn = p.name
            Hf[n] = head_at[n].copy() if n in head_at else Hf[pn] + Q[pn] @ ((b.head_local - p.head_local) * sc[pn])
            Hd[n] = Hd[pn] + Q[pn].inverted() @ (Hf[n] - Hf[pn])
            Q[n], sc[n] = Q[pn].copy(), sc[pn]
        if n in aim:
            a = aim[n]
            s_vec = (b.tail_local - b.head_local) if a.startswith(":") else (bones[a].head_local - b.head_local)
            target = tails[n] if a.startswith(":") else head_at[a]
            m_vec = target - Hf[n]
            sc[n] = m_vec.length / s_vec.length
            if n in flat:
                Q[n] = Matrix.Rotation(yaw if b.head_local.x > 0 else -yaw, 3, "Z")
            else:
                d0 = Q[n] @ s_vec.normalized()  # parent-carried direction
                Q[n] = d0.rotation_difference(m_vec.normalized()).to_matrix() @ Q[n]
        Ld[n] = b.length * sc[n]
    # stance lift: D's legs hang straight in the source's rest, the character's may stand splayed;
    # the clips' root rises by the difference so the ankles keep the character's own height
    lift = head_at["DEF-foot.L"].z - Hd["DEF-foot.L"].z
    return dict(Q=Q, Hf=Hf, Hd=Hd, scale=sc, length=Ld, head_at=head_at, tails=tails, flat=flat, lift=lift)


def build_driver(src, P):
    D = _copy_rig(src, "Driver")
    mats = {b.name: Matrix.Translation(P["Hd"][b.name]) @ b.matrix_local.to_3x3().to_4x4() for b in src.data.bones}
    _set_rest(D, mats, P["length"])
    return D


def fit_pose(D, P):
    """Pose D by pure rotations onto the landmarks; returns the max joint miss (m)."""
    _clear_anim(D)
    bones = D.data.bones
    for b in bones:
        R = b.matrix_local.to_3x3()
        Qp = P["Q"][b.parent.name] if b.parent else Matrix.Identity(3)
        pb = D.pose.bones[b.name]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (R.inverted() @ Qp.inverted() @ P["Q"][b.name] @ R).to_quaternion()
    bpy.context.view_layer.update()
    miss = 0.0
    for n, h in P["head_at"].items():
        if D.data.bones[n].parent.name not in P["flat"]:
            miss = max(miss, (D.pose.bones[n].head - h).length)
    for n, t in P["tails"].items():
        if n not in P["flat"]:
            miss = max(miss, (D.pose.bones[n].tail - t).length)
    return miss


def build_target(D, P):
    T = _copy_rig(D, "Rig")
    mats = {pb.name: pb.matrix.copy() for pb in D.pose.bones}
    _set_rest(T, mats, P["length"])
    _clear_anim(D)
    return T


# ------------------------------------------------------------------------ clips
def _rot_quat(R):
    return R.to_quaternion()


def driver_action(src_act, D, k_leg, lift, abduct):
    """A copy of the source clip for D: root/hips location x k_leg, root raised by `lift` (m, along the
    root's local z, which is world z); upper arms abducted by `abduct` (rad)."""
    act = src_act.copy()
    act.name = "D_" + src_act.name
    cb, slot = _channelbag(act)
    f0, f1 = (int(round(x)) for x in act.frame_range)
    frames = np.arange(f0, f1 + 1, dtype=np.float64)

    def sample(path, n):
        cols = []
        for i in range(n):
            fc = cb.fcurves.find(path, index=i)
            cols.append([fc.evaluate(f) for f in frames] if fc else [0.0] * len(frames))
        return np.array(cols).T

    for b in MOVING:
        path = 'pose.bones["%s"].location' % b
        v = sample(path, 3) * k_leg
        if b == "root":
            v[:, 2] += lift
        for i in range(3):
            _set_curve(cb, path, i, frames, v[:, i], b)
    if abduct:
        for s, sgn in (("L", -1.0), ("R", 1.0)):
            b = "DEF-upper_arm." + s
            R = D.data.bones[b].matrix_local.to_3x3()
            r = (R.inverted() @ Matrix.Rotation(sgn * abduct, 3, "Y") @ R).to_quaternion()
            qp, lp = 'pose.bones["%s"].rotation_quaternion' % b, 'pose.bones["%s"].location' % b
            q, loc = sample(qp, 4), sample(lp, 3)
            q2 = np.array([(r @ Quaternion(x))[:] for x in q])
            l2 = np.array([(r.to_matrix() @ Vector(x))[:] for x in loc])
            for i in range(4):
                _set_curve(cb, qp, i, frames, q2[:, i], b)
            for i in range(3):
                _set_curve(cb, lp, i, frames, l2[:, i], b)
    return act, slot, frames


def bake_clip(name, D, d_act, d_slot, frames, T):
    """T's clip: every frame, every bone's armature-space matrix = D's."""
    sc = bpy.context.scene
    _assign(D, d_act, d_slot)
    names = [b.name for b in T.data.bones]
    RT = {b.name: b.matrix_local.copy() for b in T.data.bones}
    par = {b.name: (b.parent.name if b.parent else None) for b in T.data.bones}
    loc = {n: [] for n in names}
    rot = {n: [] for n in names}
    scl = {n: [] for n in names}
    for f in frames:
        sc.frame_set(int(f))
        PD = {n: D.pose.bones[n].matrix.copy() for n in names}
        for n in names:
            p = par[n]
            B = RT[n].inverted() @ PD[n] if p is None else RT[n].inverted() @ RT[p] @ PD[p].inverted() @ PD[n]
            l, q, s = B.decompose()
            if rot[n] and q.dot(rot[n][-1]) < 0:
                q.negate()
            loc[n].append(l[:])
            rot[n].append(q)
            scl[n].append(s[:])
    act, slot, cb = _new_action("T_" + name, T.name)
    for n in names:
        L, Rq, S = np.array(loc[n]), np.array([q[:] for q in rot[n]]), np.array(scl[n])
        for i in range(3):
            _set_curve(cb, 'pose.bones["%s"].location' % n, i, frames, L[:, i], n)
            _set_curve(cb, 'pose.bones["%s"].scale' % n, i, frames, S[:, i], n)
        for i in range(4):
            _set_curve(cb, 'pose.bones["%s"].rotation_quaternion' % n, i, frames, Rq[:, i], n)
    act.use_frame_range = True
    act.frame_start, act.frame_end = frames[0], frames[-1]
    return act, slot


def verify_clip(D, d_act, d_slot, T, t_act, t_slot, frames):
    """Max over frames and bones of |head_D - head_T| (m) and of the rotation difference (rad)."""
    _assign(D, d_act, d_slot)
    _assign(T, t_act, t_slot)
    dmax = amax = 0.0
    for f in frames:
        bpy.context.scene.frame_set(int(f))
        for pb in T.pose.bones:
            pd = D.pose.bones[pb.name]
            dmax = max(dmax, (pd.head - pb.head).length, (pd.tail - pb.tail).length)
            a = pd.matrix.to_quaternion().rotation_difference(pb.matrix.to_quaternion()).angle
            amax = max(amax, min(a, 2 * math.pi - a))
    return dmax, amax


# ---------------------------------------------------------------------- weights
def label_skin(skin, sources):
    """Per skin vertex: (name of the nearest source part, distance)."""
    dg = bpy.context.evaluated_depsgraph_get()
    co = [v.co.copy() for v in skin.data.vertices]
    best = [(None, 1e9)] * len(co)
    for ob in sources:
        tree = BVHTree.FromObject(ob, dg)
        inv = ob.matrix_world.inverted()
        for i, p in enumerate(co):
            hit = tree.find_nearest(inv @ p, 0.05)
            if hit[0] is not None:
                d = (ob.matrix_world @ hit[0] - p).length
                if d < best[i][1]:
                    best[i] = (ob.name, d)
    return best


def gear_bone(part, x, gear, co=None):
    """The gear rule for a source part: None (keep the heat weights) or {bone: weight}. A rule's bone
    is a name or a {name: weight} mix -- constant weights move the gear (nearly) rigidly by a blend --
    or a callable `f(co, side)` returning such a mix (or None) for the vertex at `co`. A mix whose
    weights sum to less than 1 is PARTIAL: `bind` keeps that share of the heat weights (a rigid
    part fading into the bone-heat skin round it)."""
    if part is None:
        return None
    side = "L" if x > 0 else "R"
    for prefix, bone in gear:
        if part.startswith(prefix):
            if callable(bone):
                mix = bone(co if co is not None else (x, 0.0, 0.0), side) or {}
                mix = {b.format(s=side): w for b, w in mix.items() if w > 0.0}
                return mix or None
            if not bone:
                return None
            mix = bone if isinstance(bone, dict) else {bone: 1.0}
            return {b.format(s=side): w for b, w in mix.items()}
    return None


def bind(skin, T, labels, gear):
    """Bone heat at T's rest, then rigid gear, at most 4 influences, normalised."""
    names = [b.name for b in T.data.bones]
    idx = {n: i for i, n in enumerate(names)}
    T.data.pose_position = "REST"
    T.data.bones["root"].use_deform = False  # lies on the floor behind the heels: no heat for it
    skin.vertex_groups.clear()
    for m in list(skin.modifiers):
        skin.modifiers.remove(m)
    gameready._select([skin, T], T)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    T.data.bones["root"].use_deform = True
    me = skin.data
    W = np.zeros((len(me.vertices), len(names)))
    gname = {g.index: g.name for g in skin.vertex_groups}
    for v in me.vertices:
        for g in v.groups:
            if gname[g.group] in idx:
                W[v.index, idx[gname[g.group]]] = g.weight
    heat_empty = int((W.sum(1) <= 1e-6).sum())
    rigid = {}
    for v in me.vertices:
        mix = gear_bone(labels[v.index][0], v.co.x, gear, v.co)
        if mix:
            share = sum(mix.values())
            if share >= 1.0 - 1e-9:
                W[v.index] = 0.0
                for b, w in mix.items():
                    W[v.index, idx[b]] = w
                key = "+".join(sorted(mix))
            else:  # partial: the heat weights keep the rest
                h = W[v.index].sum()
                W[v.index] *= (1.0 - share) / h if h > 1e-9 else 0.0
                for b, w in mix.items():
                    W[v.index, idx[b]] += w
                key = "+".join(sorted(mix)) + " (partial)"
            rigid[key] = rigid.get(key, 0) + 1
    # a vertex the heat solve left empty takes its nearest bone segment
    empty = np.flatnonzero(W.sum(1) <= 1e-6)
    for i in empty:
        p = me.vertices[i].co
        d = [_seg_dist(p, b.head_local, b.tail_local) if b.name != "root" else 1e9 for b in T.data.bones]
        W[i, int(np.argmin(d))] = 1.0
    keep = np.argsort(-W, axis=1)[:, :4]
    W4 = np.zeros_like(W)
    rows = np.arange(len(W))[:, None]
    W4[rows, keep] = W[rows, keep]
    W4 /= W4.sum(1, keepdims=True)
    skin.vertex_groups.clear()
    for j, n in enumerate(names):
        vs = np.flatnonzero(W4[:, j] > 1e-6)
        vg = skin.vertex_groups.new(name=n)
        for i in vs:
            vg.add([int(i)], float(W4[i, j]), "REPLACE")
    T.data.pose_position = "POSE"
    infl = (W4 > 1e-6).sum(1)
    return dict(heat_empty_vertices=heat_empty, rigid_vertices=dict(sorted(rigid.items())),
                influences_max=int(infl.max()), influences_mean=round(float(infl.mean()), 2),
                vertices_over_4_before_limit=int(((W > 1e-6).sum(1) > 4).sum()))


def _seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-12)))
    return (a + ab * t - p).length


# ------------------------------------------------------------------ floor clamp
CLAMP_STEP_DEG, CLAMP_MAX_DEG = 0.5, 45.0


def _pitch_search(pts_local, P, pivot, axis):
    """Smallest rotation (rad) about `axis` through `pivot` lifting every point (bone-local, posed by
    P) to z >= 0; if none reaches it, the one with the least penetration. Returns (angle, min z after)."""
    W = np.array([(P @ Vector(p))[:] for p in pts_local])
    z0 = W[:, 2].min()
    if z0 >= 0.0:
        return 0.0, z0
    rel = W - np.array(pivot[:])
    a = np.array(axis[:])
    best = (z0, 0.0)
    steps = np.arange(CLAMP_STEP_DEG, CLAMP_MAX_DEG + 1e-9, CLAMP_STEP_DEG)
    for deg in steps:
        for th in (math.radians(deg), -math.radians(deg)):
            c, s = math.cos(th), math.sin(th)
            # Rodrigues, z component only
            z = pivot[2] + rel[:, 2] * c + np.cross(a, rel)[:, 2] * s + a[2] * (rel @ a) * (1 - c)
            m = z.min()
            if m >= 0.0:
                return th, m
            if m > best[0]:
                best = (m, th)
    return best[1], best[0]


def floor_clamp(T, skin, clips, frames_by_clip):
    """Keep the boots out of the floor: per clip frame, the foot bone is pitched about the ankle by
    the smallest angle that lifts every vertex bound mainly to it to z >= 0 (a heel strike's heel, a
    toe-off's ball), then the toe bone about the ball for the vertices bound to it. Only DEF-foot.* and
    DEF-toe.* keys change. Returns per-clip stats (largest correction, worst penetration left, mm)."""
    sc = bpy.context.scene
    names = {g.index: g.name for g in skin.vertex_groups}
    RT = {b.name: b.matrix_local.copy() for b in T.data.bones}
    sets = {}
    for s in ("L", "R"):
        for b in ("DEF-foot." + s, "DEF-toe." + s):
            inv = RT[b].inverted()
            sets[b] = [(inv @ v.co)[:] for v in skin.data.vertices
                       if any(names[g.group] == b and g.weight > 0.5 for g in v.groups)]
    out = {}
    for clip, (act, slot) in clips.items():
        _assign(T, act, slot)
        frames = frames_by_clip[clip]
        cb, _ = _channelbag(act)
        keys = {}
        worst_fix = worst_left = 0.0
        for f in frames:
            sc.frame_set(int(f))
            for s in ("L", "R"):
                fn, tn, sn = "DEF-foot." + s, "DEF-toe." + s, "DEF-shin." + s
                Pf, Pt, Ps = (T.pose.bones[n].matrix.copy() for n in (fn, tn, sn))
                th, _m = _pitch_search(sets[fn], Pf, Pf.translation, Pf.col[0].xyz.normalized())
                R1 = Matrix.Translation(Pf.translation) @ Matrix.Rotation(th, 4, Pf.col[0].xyz.normalized()) \
                    @ Matrix.Translation(-Pf.translation)
                Pf2, Pt2 = R1 @ Pf, R1 @ Pt
                th2, m2 = _pitch_search(sets[tn], Pt2, Pt2.translation, Pt2.col[0].xyz.normalized())
                R2 = Matrix.Translation(Pt2.translation) @ Matrix.Rotation(th2, 4, Pt2.col[0].xyz.normalized()) \
                    @ Matrix.Translation(-Pt2.translation)
                Pt2 = R2 @ Pt2
                m1 = min((Pf2 @ Vector(p)).z for p in sets[fn]) if sets[fn] else 0.0
                worst_fix = max(worst_fix, abs(math.degrees(th)), abs(math.degrees(th2)))
                worst_left = min(worst_left, m1, m2)
                keys.setdefault(fn, []).append(RT[fn].inverted() @ RT[sn] @ Ps.inverted() @ Pf2)
                keys.setdefault(tn, []).append(RT[tn].inverted() @ RT[fn] @ Pf2.inverted() @ Pt2)
        for n, mats in keys.items():
            q_prev, Q = None, []
            for M in mats:
                q = M.to_quaternion()
                if q_prev is not None and q.dot(q_prev) < 0:
                    q.negate()
                Q.append(q[:])
                q_prev = q
            Q = np.array(Q)
            for i in range(4):
                _set_curve(cb, 'pose.bones["%s"].rotation_quaternion' % n, i, frames, Q[:, i], n)
            L = np.array([M.to_translation()[:] for M in mats])
            for i in range(3):
                _set_curve(cb, 'pose.bones["%s"].location' % n, i, frames, L[:, i], n)
        out[clip] = dict(max_correction_deg=round(worst_fix, 1), penetration_left_mm=round(worst_left * 1e3, 1))
    _clear_anim(T)
    return out


# --------------------------------------------------------------------- previews
def _camera(name, loc, target, ortho=None, lens=50):
    cd = bpy.data.cameras.new(name)
    if ortho:
        cd.type, cd.ortho_scale = "ORTHO", ortho
    else:
        cd.lens = lens
    cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return cam


def _floor():
    me = bpy.data.meshes.new("_floor")
    s = 3.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("_floor", me)
    bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new("_floor")
    m.diffuse_color = (0.30, 0.31, 0.33, 1.0)
    me.materials.append(m)
    return ob


def render_poses(T, skin, clips, out_png, tmp_dir, px=(420, 560), poses=TEST_POSES):
    """3/4 view (top row) and straight down (bottom row) per test pose; returns per-pose stats."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type, sh.show_shadows, sh.show_cavity = "STUDIO", "TEXTURE", True, True
    sc.display.shadow_shift = 0.1
    sc.render.resolution_x, sc.render.resolution_y = px
    sc.render.film_transparent = False
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.color = (0.36, 0.375, 0.41)
    sc.render.image_settings.file_format = "PNG"
    floor = _floor()
    tiles, info = [], []
    for clip, frame in poses:
        act, slot = clips[clip]
        _assign(T, act, slot)
        f = int(act.frame_end) if frame < 0 else frame
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        me = skin.evaluated_get(dg).to_mesh()
        co = np.array([v.co[:] for v in me.vertices])
        skin.evaluated_get(dg).to_mesh_clear()
        c = (co.min(0) + co.max(0)) / 2
        info.append(dict(clip=clip, frame=f, min_z_m=round(float(co[:, 2].min()), 4),
                         height_m=round(float(co[:, 2].max() - co[:, 2].min()), 3)))
        col = []
        for view in ("q", "top"):
            if view == "q":
                tgt = (c[0], c[1], max(c[2], 0.5))
                cam = _camera("_cam", (c[0] - 2.4, c[1] - 3.4, 1.9), tgt, lens=50)
            else:
                cam = _camera("_cam", (c[0], c[1], 6.0), (c[0], c[1], 0.0), ortho=2.4)
                sc.render.resolution_x, sc.render.resolution_y = px[0], px[0]
            sc.camera = cam
            path = os.path.join(tmp_dir, "_pose_%s_%d_%s.png" % (clip, f, view))
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            sc.render.resolution_x, sc.render.resolution_y = px
            bpy.data.objects.remove(cam)
            img = bpy.data.images.load(path)
            w, h = img.size
            a = np.empty(w * h * 4, np.float32)
            img.pixels.foreach_get(a)
            bpy.data.images.remove(img)
            os.remove(path)
            col.append(a.reshape(h, w, 4))
        tiles.append(col)
    bpy.data.objects.remove(floor)
    gap = 6
    W = sum(t[0].shape[1] for t in tiles) + gap * (len(tiles) - 1)
    H = tiles[0][0].shape[0] + gap + tiles[0][1].shape[0]
    sheet = np.ones((H, W, 4), np.float32)
    x = 0
    for t in tiles:  # Blender rows: 0 = bottom -> the top view goes first (bottom), the 3/4 view above it
        q, top = t
        sheet[:top.shape[0], x:x + top.shape[1]] = top
        sheet[top.shape[0] + gap:, x:x + q.shape[1]] = q
        x += q.shape[1] + gap
    gameready._save_png(sheet, out_png, "sRGB")
    return info


# -------------------------------------------------------------------------- run
def _offset(skin, game, name):
    if "export_offset" in skin:
        return Vector(skin["export_offset"])
    with open(os.path.join(game, "stats.json")) as f:
        return Vector(json.load(f)["export"]["offset_m"])


def run(args, root, name, build, make_materials, spec, out_glb, previews_dir=None):
    t0 = time.time()
    timing = {}
    game, prev = os.path.join(root, "game"), previews_dir or os.path.join(root, "previews")
    os.makedirs(prev, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(game, name + "_game.blend"))
    sc = bpy.context.scene
    sc.render.fps, sc.render.fps_base = 24, 1.0
    skin = bpy.data.objects["GameSkin"]
    for ob in list(bpy.data.objects):
        if ob is not skin:
            bpy.data.objects.remove(ob)
    # back into the character's own frame (P1 moved the export by the feet's mean x/y)
    off = _offset(skin, game, name)
    skin.data.transform(Matrix.Translation(-off))
    skin.matrix_world = Matrix.Identity(4)

    src, src_actions = import_source()
    P = plan(src, spec)
    D = build_driver(src, P)
    miss = fit_pose(D, P)
    T = build_target(D, P)
    bpy.data.objects.remove(src)
    T.name = T.data.name = "Rig"
    timing["rigs_s"] = round(time.time() - t0, 1)
    print("rig: fit-pose joint miss %.4f mm, stance lift %.1f mm" % (miss * 1e3, P["lift"] * 1e3), flush=True)

    # leg-length ratio: the root/hips translation scale
    leg = lambda s: sum(P["length"]["DEF-%s.L" % b] / (P["scale"]["DEF-%s.L" % b] if s else 1.0) for b in ("thigh", "shin"))
    k_leg = leg(False) / leg(True)
    abduct = math.radians(getattr(args, "abduct", None) if getattr(args, "abduct", None) is not None
                          else spec.ABDUCTION_DEG)

    t = time.time()
    clips, d_clips, exact = {}, {}, {}
    for a in src_actions:
        d_act, d_slot, frames = driver_action(a, D, k_leg, P["lift"], abduct)
        t_act, t_slot = bake_clip(a.name, D, d_act, d_slot, frames, T)
        clips[a.name] = (t_act, t_slot)
        d_clips[a.name] = (d_act, d_slot, frames)
    timing["clips_s"] = round(time.time() - t, 1)
    t = time.time()
    for n, (d_act, d_slot, frames) in d_clips.items():
        dm, am = verify_clip(D, d_act, d_slot, T, *clips[n], frames)
        exact[n] = (dm, am)
    timing["verify_s"] = round(time.time() - t, 1)
    worst = max(exact, key=lambda n: exact[n][0])
    print("rig: %d clips, worst D-T head/tail distance %.6f mm (%s), worst angle %.2e rad"
          % (len(clips), exact[worst][0] * 1e3, worst, max(v[1] for v in exact.values())), flush=True)
    _clear_anim(D)
    _clear_anim(T)

    # weights: label the skin by the high-res parts, bone heat, rigid gear
    t = time.time()
    keep = set(bpy.data.objects)
    kit.RES = 0.004
    build(make_materials())
    sources = [o for o in bpy.data.objects if o not in keep and o.type == "MESH"]
    labels = label_skin(skin, sources)
    for ob in [o for o in bpy.data.objects if o not in keep]:
        bpy.data.objects.remove(ob)
    unlabelled = sum(1 for l in labels if l[0] is None)
    timing["label_s"] = round(time.time() - t, 1)
    t = time.time()
    wstats = bind(skin, T, labels, spec.GEAR)
    wstats["unlabelled_vertices"] = unlabelled
    timing["bind_s"] = round(time.time() - t, 1)
    print("rig: weights", json.dumps(wstats), flush=True)
    clamp = None
    if getattr(spec, "FLOOR_CLAMP", False):  # opt-in: the boots kept out of the floor
        t = time.time()
        clamp = floor_clamp(T, skin, clips, {n: v[2] for n, v in d_clips.items()})
        timing["floor_clamp_s"] = round(time.time() - t, 1)
        print("rig: floor clamp", json.dumps(clamp), flush=True)

    t = time.time()
    poses = []
    for m in skin.data.materials:  # the Workbench shows the ACTIVE image node: the albedo
        tex = [n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image.colorspace_settings.name == "sRGB"]
        if tex:
            m.node_tree.nodes.active = tex[0]
    if not args.no_previews:
        poses = render_poses(T, skin, clips, os.path.join(prev, "rig_poses.png"), prev,
                             poses=getattr(spec, "TEST_POSES", TEST_POSES))
        print("rig: poses", json.dumps(poses), flush=True)
    timing["previews_s"] = round(time.time() - t, 1)

    # export: T + skin only, the clips under their own names
    t = time.time()
    bpy.data.objects.remove(D)
    for a in list(bpy.data.actions):
        if not a.name.startswith("T_"):
            bpy.data.actions.remove(a)
    _clear_anim(T)
    for n, (act, slot) in sorted(clips.items()):
        act.name = n
        act.use_fake_user = True
        tr = T.animation_data.nla_tracks.new()
        tr.name = n
        st = tr.strips.new(n, int(act.frame_start), act)
        st.action_slot = slot
        tr.mute = True
    skin.name = skin.data.name = name
    os.makedirs(os.path.dirname(out_glb), exist_ok=True)
    gameready._select([skin, T], T)
    bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=False, export_texcoords=True, export_normals=True, export_tangents=True,
                              export_materials="EXPORT", export_image_format="AUTO", export_skins=True,
                              export_all_influences=False, export_def_bones=False, export_rest_position_armature=True,
                              export_animations=True, export_animation_mode="ACTIONS", export_force_sampling=True,
                              export_frame_step=1, export_anim_single_armature=True, export_reset_pose_bones=True,
                              export_morph=False, export_cameras=False, export_lights=False)
    gameready._splice_albedo(out_glb, os.path.join(game, "albedo.png"))  # the gloss mask in its alpha (#33)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(game, name + "_rigged.blend"))
    timing["export_s"] = round(time.time() - t, 1)
    timing["total_s"] = round(time.time() - t0, 1)

    gl = _gltf_summary(out_glb)
    stats_path = os.path.join(game, "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    ex_mm = {n: round(v[0] * 1e3, 6) for n, v in exact.items()}
    stats["rig"] = dict(
        method="driver rig D (source re-proportioned, T-pose) -> fit pose -> target rig T; clips baked by world matrix",
        source_rig=os.path.relpath(SOURCE_RIG, REPO).replace("\\", "/"),
        bones=len(T.data.bones), clips=len(clips),
        fit_pose_joint_miss_mm=round(miss * 1e3, 4),
        leg_length_ratio=round(k_leg, 4), stance_lift_mm=round(P["lift"] * 1e3, 1),
        foot_yaw_deg=getattr(spec, "FOOT_YAW_DEG", 0.0), abduction_deg=round(math.degrees(abduct), 2),
        bone_scale={n: round(v, 3) for n, v in P["scale"].items() if n in P["head_at"] or n in _sides(AIM)},
        clip_exactness_max_head_or_tail_mm=dict(worst_clip=worst, worst=max(ex_mm.values()),
                                                idle=ex_mm.get("Idle_Loop"), walk=ex_mm.get("Walk_Loop"),
                                                death=ex_mm.get("Death01"), pistol_shoot=ex_mm.get("Pistol_Shoot")),
        clip_exactness_max_angle_rad=float("%.3g" % max(v[1] for v in exact.values())),
        weights=wstats, test_poses=poses,
        **(dict(floor_clamp=dict(max_correction_deg=max(v["max_correction_deg"] for v in clamp.values()),
                                 worst_penetration_left_mm=min(v["penetration_left_mm"] for v in clamp.values()),
                                 per_clip=clamp)) if clamp else {}), offset_removed_m=[round(x, 4) for x in off],
        export=dict(file=os.path.relpath(out_glb, REPO).replace("\\", "/"), bytes=os.path.getsize(out_glb), **gl),
        timing_s=timing)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
    print("rig stats:", json.dumps(stats["rig"]))
    print("rig done in %.1fs" % (time.time() - t0))


def _gltf_summary(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[12:16], "little")
    g = json.loads(data[20:20 + n])
    return dict(meshes=len(g.get("meshes", [])), skins=len(g.get("skins", [])),
                joints=len(g["skins"][0]["joints"]) if g.get("skins") else 0,
                animations=len(g.get("animations", [])), materials=len(g.get("materials", [])),
                images=len(g.get("images", [])))
