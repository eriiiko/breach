"""Blender 4.5: the mid-size body-plan options beside the game's marine, at true scale (#106).

blender -b --factory-startup -P build_bodyplan.py -- --root <checkout> [--only mid,mid_bulk] [--samples 48]

For each option row of octopus_math.SIZES (A mid_bulk, B mid = the default,
C mid_heavy) the hero octopus is assembled at that row's dials, given the
build's own materials and studio (build_octopus), and rendered at rest and
reaching beside the space marine (assets/models/space_marine, the scale every
game look shares), from straight above (the game camera) and three-quarter.
Every option uses the SAME cameras, so the pictures compare sizes directly.
Also renders the stretch strip (one arm at stretch 0 / 0.5 / 1, default row).
Writes PNGs to previews/bodyplan/raw/ and bodyplan.json (the cameras' metres per
pixel, for the labelled sheets made by package_bodyplan.py). Render-only.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_octopus as bo
import octopus_math as om

OUT = bo.HOME / "previews" / "bodyplan"
RAW = OUT / "raw"
OPTIONS = ["mid_bulk", "mid", "mid_heavy"]          # A, B (default), C
# The marine stands on the animal's right (-X), facing it; the reaching arms are R2 and R3.
MARINE_AT = np.array([-2.85, 0.0, 0.15])            # glTF metres (crown axis at the origin)
TOP_CENTRE = np.array([-0.45, 0.0, 0.0])
TOP_ORTHO, TOP_RES = 7.6, (1200, 1000)
CAM34 = dict(loc=(3.2, 5.6, 5.6), target=(-0.8, 0.35, -0.1), lens=30, res=(1400, 1000))
STRIP_RES = (900, 1100)
LIGHT_SCALE = 2.0                                   # the build studio, sized for a ~2x scene (same look)


def reach_arms():
    """Rest arms, with R2 and R3 fully stretched out toward the marine (on the animal's right, -X)."""
    arms = om.rest_arms()
    i2, i3 = om.ARM_NAMES.index("R2"), om.ARM_NAMES.index("R3")
    # opposite bend directions curve the two tips to either side of him (tips ~0.6 m apart)
    arms[i2] = om.ArmParams(curl=0.04, bend_dir=-0.5, twist=0.0, tip_curl=0.35, lift=0.20, swing=0.0, stretch=1.0)
    arms[i3] = om.ArmParams(curl=0.04, bend_dir=0.5, twist=0.0, tip_curl=0.35, lift=0.20, swing=0.0, stretch=1.0)
    return arms


def strip_arms(stretch):
    arms = om.rest_arms()
    arms[0] = om.ArmParams(curl=0.0, bend_dir=0.0, tip_curl=0.0, lift=0.05, swing=0.0, stretch=stretch)
    return arms


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.world = bpy.data.worlds.new("World")
    bo.gpu()


def import_marine(root):
    path = root / "assets" / "models" / "space_marine" / "space_marine.glb"
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    objs = [o for o in bpy.data.objects if o not in before]
    rig = next((o for o in objs if o.type == "ARMATURE"), None)
    if rig is not None:
        idle = bpy.data.actions.get("Idle_Loop")                       # the game's idle clip (CLIP_MAP)
        if idle is not None:
            rig.animation_data_create()
            rig.animation_data.action = idle
        roots = [o for o in objs if o.parent is None]
    else:
        roots = [o for o in objs if o.parent is None]
    for o in roots:
        o.location = Vector(bo.to_b(MARINE_AT))
        o.rotation_mode = "XYZ"
        o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1], math.radians(90))   # glTF +Z forward -> face +X (the octopus)
    bpy.context.scene.frame_set(10)
    return objs


def measure(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for o in objs:
        if o.type != "MESH":
            continue
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        co = np.zeros(len(m.vertices) * 3)
        m.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(o.matrix_world)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        lo, hi = np.minimum(lo, co.min(0)), np.maximum(hi, co.max(0))
        ev.to_mesh_clear()
    return lo, hi


def build(name):
    om.set_size(name)
    t0 = time.time()
    hero = om.assemble("hero")
    bo.log(name, "assembled", round(time.time() - t0, 1), "s")
    obj = bo.build_mesh(hero, f"Octopus_{name}", [bo.skin_material(bo.skin_mapping_scale(hero["vertices"])),
                                                  bo.cup_material(), bo.eye_material()])
    rig = bo.build_rig(hero, f"Octopus_{name}_rig", obj)
    return obj, rig


def top(cam, path, centre=TOP_CENTRE, ortho=TOP_ORTHO, res=TOP_RES):
    c = np.asarray(centre, float)
    bo.shot(cam, path, bo.to_b(c + [0, 12, 0]), bo.to_b(c), ortho=ortho, res=res)
    # The camera's up is Blender +Y (glTF -Z, the animal's back); record the mapping.
    return dict(centre_gltf=c.tolist(), ortho_m=ortho, res=list(res), m_per_px=ortho / max(res))


def three_quarter(cam, path):
    bo.shot(cam, path, bo.to_b(CAM34["loc"]), bo.to_b(CAM34["target"]), lens=CAM34["lens"], res=CAM34["res"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--no-strip", action="store_true")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    RAW.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "bodyplan.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["marine_at_gltf_m"] = MARINE_AT.tolist()
    names = [n for n in OPTIONS if not args.only or n in args.only.split(",")]
    for name in names:
        clear()
        obj, rig = build(name)
        cam, floor = bo.studio(LIGHT_SCALE)
        bpy.context.scene.cycles.samples = args.samples
        marine = import_marine(args.root.resolve())
        lo, hi = measure(marine)
        rec = dict(size=om.size_report(name), marine_height_m=float(hi[2] - lo[2]),
                   marine_bbox_blender=[lo.tolist(), hi.tolist()])
        for pose, arms in (("rest", om.rest_arms()), ("reach", reach_arms())):
            bo.set_pose(rig, om.BodyParams(), arms, "hero")
            rec[f"{pose}_top"] = top(cam, RAW / f"{name}_{pose}_top.png")
            three_quarter(cam, RAW / f"{name}_{pose}_34.png")
            # closest approach of the octopus skin to the marine's vertical axis (no intersection)
            v = bo.evaluated_vertices(obj)
            d = np.hypot(v[:, 0] - MARINE_AT[0], v[:, 2] - MARINE_AT[2])
            rec[f"{pose}_closest_to_marine_axis_m"] = float(d.min())
        report.setdefault("options", {})[name] = rec
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        bo.log("BODYPLAN option", name, json.dumps({k: v for k, v in rec.items() if "closest" in k}))
    if not args.no_strip:
        clear()
        obj, rig = build("mid")
        cam, floor = bo.studio(LIGHT_SCALE)
        bpy.context.scene.cycles.samples = args.samples
        strip = {}
        for st in (0.0, 0.5, 1.0):
            bo.set_pose(rig, om.BodyParams(), strip_arms(st), "hero")
            k = 0
            tip = om.pose_octopus(om.BodyParams(), strip_arms(st), "hero")[0][om.arm_bone_offset(k) + 32]
            # frame the body and arm L1 (front-left): same camera for the three values
            strip[str(st)] = top(cam, RAW / f"strip_{st:.1f}.png", centre=(0.75, 0, 0.95), ortho=3.9,
                                 res=STRIP_RES)
            strip[str(st)]["arm_length_m"] = om.arm_length(st) * om.OCTOPUS_SCALE
            strip[str(st)]["base_to_tip_m"] = float(np.linalg.norm(tip - om.MOUNT_POINTS[k] * om.OCTOPUS_SCALE))
            strip[str(st)]["thinning"] = float(om.arm_joint_scales(st)[-1])
        report["strip"] = strip
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    bo.log("BODYPLAN_OK", names)


if __name__ == "__main__":
    main()
