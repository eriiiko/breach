"""Beatrice's face pictures (B2a): the saved build (`source/Beatrice.blend`) opened and its head rendered, never saved.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py -- [options]
    (from art/characters; then `python Beatrice/scripts/face_sheets.py` lays out face_vs_art.jpg, face_turn.jpg, face_closeup.jpg)

    --blend PATH     another saved build
    --head NAME      rebuild the head on the opened build from `beatrice.head(NAME)` first (e.g. `concept`, the head-size variant)
    --only a,b       art, turn, close (default all)
    --samples N      Cycles samples (default 96)
    --out DIR        where the raw RGBA renders go (default $TEMP/Beatrice_face)

The shots are `beatrice.FACE_SHOTS`; the turn and the close-ups use the optional portrait light (`POSE["lights"]["portrait"]`),
the art matches the default studio rig.
"""
import argparse
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import bpy  # noqa: E402
import beatrice  # noqa: E402
import studio  # noqa: E402
import suitpose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--blend", default="")
ap.add_argument("--head", default="")
ap.add_argument("--only", default="art,turn,close")
ap.add_argument("--samples", type=int, default=96)
ap.add_argument("--out", default="")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
root = os.path.normpath(os.path.join(HERE, ".."))
bpy.ops.wm.open_mainfile(filepath=args.blend or os.path.join(root, "source", "Beatrice.blend"))
sc = bpy.context.scene
sc.cycles.samples = args.samples
print("render device:", suitpose.use_gpu())
if args.head:
    import animehead
    spec = beatrice.head(args.head)
    for ob in list(bpy.data.collections["Head"].objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    mats = animehead.materials("beatrice", beatrice.palette())
    animehead.build("Beatrice", spec["face"], spec["eyes"], mats, "Head", h=spec.get("res", 0.0007), hair=spec.get("hair"),
                    scale=spec.get("scale", 1.0))
out = args.out or os.path.join(tempfile.gettempdir(), "Beatrice_face")
os.makedirs(out, exist_ok=True)
tag = (args.head + "_") if args.head else ""
cam, rig = bpy.data.objects["Cam"], bpy.data.objects["StudioRig"]
floor = bpy.data.objects.get("Floor")
if floor:
    floor.hide_render = True
hair = [o for o in bpy.data.objects if o.name.startswith("Beatrice_Hair")]
S = beatrice.FACE_SHOTS
only = args.only.split(",")

if "art" in only:
    cam.data.type = "ORTHO"
    cam.data.clip_start, cam.data.clip_end = 0.01, 40.0
    for name, s in S["art"].items():
        cam.data.ortho_scale = s["width"]
        u, z = s["centre"]
        if s["side"] == "front":
            cam.location, cam.rotation_euler = (u, -3.0, z), (math.radians(90.0), 0.0, 0.0)
        else:   # from her right: image right is her front
            cam.location, cam.rotation_euler = (-3.0, u, z), (math.radians(90.0), 0.0, math.radians(-90.0))
        for with_hair in (True, False):
            for o in hair:
                o.hide_render = not with_hair
            studio.render(os.path.join(out, "%sart_%s%s.png" % (tag, name, "" if with_hair else "_nohair")), (1254, 1254), transparent=True)
    for o in hair:
        o.hide_render = False

suitpose.add_lights(beatrice.POSE["lights"]["portrait"], rig)
if "turn" in only:
    T = S["turn"]
    cam.data.type = "PERSP"
    cam.data.lens = T["lens"]
    tx, ty, tz = T["target"]
    for az in T["azimuths"]:
        a, e = math.radians(az), math.radians(T["elev"])
        cam.location = (tx + T["dist"] * math.sin(a) * math.cos(e), ty - T["dist"] * math.cos(a) * math.cos(e), tz + T["dist"] * math.sin(e))
        studio._look_at(cam, T["target"])
        studio.render(os.path.join(out, "%sturn_%03d.png" % (tag, az)), T["res"])
if "close" in only:
    cam.data.type = "PERSP"
    for name, s in S["close"].items():
        cam.data.lens = s["lens"]
        cam.location = s["cam"]
        studio._look_at(cam, s["target"])
        studio.render(os.path.join(out, "%sclose_%s.png" % (tag, name)), s["res"])
print("face views ->", out)
