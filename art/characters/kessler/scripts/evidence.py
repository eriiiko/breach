"""Kessler's game-ready picture set, from the saved rigged file (`game.py` writes game/kessler_rigged.blend):

    blender -b --factory-startup -P art/characters/kessler/scripts/evidence.py

Into previews/game/: the test poses and extreme poses (front three-quarter + straight down, the
worker kit's `workergame.shoot_poses`), helmet and hand close-ups per pose (`shoot_faces`), the
helmet at rest, turned and nodded (the head's rigid block must not tear at the neck), and the
Walk_Loop floor check; numbers into game/stats.json["evidence"].
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import workergame as wg  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
PREV = os.path.join(ROOT, "previews", "game")
POSES = (("Idle_Loop", 0), ("Walk_Loop", "stride"), ("Walk_Loop", 16), ("Pistol_Shoot", 5), ("Death01", -1),
         ("Crouch_Idle_Loop", 0))
EXTREMES = (("Sword_Attack", 14), ("Fixing_Kneeling", 40), ("Sitting_Idle_Loop", 0), ("Jump_Loop", 6))
# the head poses: (name, neck yaw, head yaw, neck pitch, head pitch), degrees; + yaw turns to his left
HEAD_POSES = (("rest", 0, 0, 0, 0), ("turn_left", 15, 30, 0, 0), ("turn_right", -15, -30, 0, 0), ("nod", 0, 0, 10, 20))


def _rot_local(T, name, axis, deg):
    """Pose bone `name` by `deg` about the ARMATURE axis `axis` (from its rest; parents at rest)."""
    pb = T.pose.bones[name]
    B = T.data.bones[name].matrix_local.to_3x3()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (B.inverted() @ Matrix.Rotation(math.radians(deg), 3, axis) @ B).to_quaternion() \
        @ pb.rotation_quaternion


def shoot_helmet(T, skin):
    import rig
    out = []
    for name, ny, hy, npitch, hp in HEAD_POSES:
        rig._clear_anim(T)
        _rot_local(T, "DEF-neck", "Z", ny)
        _rot_local(T, "DEF-head", "Z", hy)
        _rot_local(T, "DEF-neck", "X", npitch)
        _rot_local(T, "DEF-head", "X", hp)
        bpy.context.view_layer.update()
        hb = T.pose.bones["DEF-head"]
        c = hb.head + Vector((0.0, -0.02, 0.08))
        for view, loc in (("front", c + Vector((0.55, -0.95, 0.20))), ("side", c + Vector((1.05, -0.15, 0.10))),
                          ("top", c + Vector((0.0, -0.05, 1.6)))):
            p = os.path.join(PREV, "helmet_%s_%s.jpg" % (name, view))
            wg._shoot(loc, c, p, (620, 620))
            out.append(os.path.basename(p))
    rig._clear_anim(T)
    return out


def main():
    t0 = time.time()
    T, skin = wg._open_rigged(ROOT, "kessler")
    wg._workbench((640, 820))
    # the Workbench's cavity shading paints the AI mesh's many small ridges white: off, so the
    # pictures show the baked texture as the game draws it
    bpy.context.scene.display.shading.show_cavity = False
    # the albedo's alpha is the GLOSS mask (#33), which the game never reads as coverage; the
    # Workbench would draw it as hashed transparency (white holes wherever the armour is matte)
    for m in skin.data.materials:
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image.colorspace_settings.name == "sRGB":
                n.image.alpha_mode = "NONE"
    os.makedirs(PREV, exist_ok=True)
    poses = wg.shoot_poses(T, skin, POSES, PREV, "pose_")
    extremes = wg.shoot_poses(T, skin, EXTREMES, PREV, "extreme_")
    faces = wg.shoot_faces(T, skin, PREV)
    helmet = shoot_helmet(T, skin)
    walk = wg.walk_floor(T, skin)
    import rig  # the rig stage's own pose sheet again, without the alpha holes
    import rig_spec
    clips = {a.name: (a, a.slots[0]) for a in bpy.data.actions}
    rig.render_poses(T, skin, clips, os.path.join(PREV, "rig_poses.png"), PREV, poses=rig_spec.TEST_POSES)
    path = os.path.join(ROOT, "game", "stats.json")
    with open(path) as f:
        stats = json.load(f)
    stats["evidence"] = dict(walk_loop_floor=walk, poses=poses, extremes=extremes, close_ups=faces, helmet=helmet,
                             timing_s=round(time.time() - t0, 1))
    with open(path, "w") as f:
        json.dump(stats, f, indent=1)
    print("evidence:", json.dumps(stats["evidence"]), flush=True)


main()
